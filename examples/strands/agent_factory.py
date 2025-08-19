import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple
import re

from dotenv import load_dotenv, find_dotenv
from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools.browser import LocalChromiumBrowser
from strands_tools import http_request, file_read, calculator, python_repl
from pydantic import ValidationError
try:
    from strands_tools.browser.models import BrowserInput
except ImportError:
    BrowserInput = None


SYSTEM_PROMPT = """
You are a research agent. Always ground answers in tool-based evidence. Do not guess.

Global loop (repeat until confident):
1) Thought: declare Schema=[minimal fields], focus_dim=<time|geo|definition|number>, and ToolPlan (which tool and why).
2) Action: execute the best tool for the step. Prefer tool_calls if available. Otherwise, output the strict 2-line format.

Tool selection rules:
- browser: general navigation and reading.
  - First action: init_session with a kebab-case session_name (e.g., "research-1"); reuse it.
  - After navigate: usually read with get_text  (e.g., selectors "body", "main", "h1,h2,h3", "p"), if you need to interact, use click items or type things. Avoid looping the same read on a page.
  - If you are stucked by Captcha, use https://duckduckgo.com to do the search instead.
- http_request: prefer for structured/fast sources (GitHub/Wikipedia/USGS APIs or light HTML).
- file_read: open local/remote files (PDF/CSV/text). For PDFs, extract essential text snippets.
- calculator: arithmetic/unit conversions; for complex code use python_repl.

Answer policy:
- Only output final_answer after at least one Observation supports it (prefer two). Keep answers concise.
- If the current approach fails twice, change strategy (different query/tool) rather than repeating the same navigate/get_text.

STRICT OUTPUT (non-tool_call mode; EXACTLY two lines per step):
Thought: <Schema=[...]; focus_dim=...; ToolPlan=...; reasoning>
Action: {"name":"browser|http_request|file_read|calculator|python_repl|final_answer","arguments":{...}}
"""

USER_TEMPLATE = """Question: {question}
History: {history}
{schema_hint}

Output EXACTLY 2 lines:
Thought: [analysis]
Action: [JSON]"""

ACTION_RE = re.compile("Action:\\s*(\\{.*\\})", re.DOTALL | re.IGNORECASE)
SCHEMA_RE = re.compile("Schema\\s*:\\s*(\\[[^\\]]*\\])", re.IGNORECASE)

def parse_schema(text: str) -> Optional[list]:
    m = SCHEMA_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        raw = m.group(1).strip("[] ")
        if not raw:
            return None
        return [x.strip() for x in raw.split(",") if x.strip()]


def parse_action(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    thought = ""
    m_thought = re.search(r"Thought:(.*?)(?:\nAction:|$)", text, re.DOTALL | re.IGNORECASE)
    if m_thought:
        thought = m_thought.group(1).strip()
    m = ACTION_RE.search(text.strip())
    if not m:
        return None, thought
    try:
        return json.loads(m.group(1)), thought
    except Exception:
        return None, thought


def build_llm_tool_specs() -> list[dict[str, Any]]:
    """Build a unified set of OpenAI-style tool specs for function calling.

    Includes: browser, http_request, file_read, calculator, python_repl, final_answer.
    Falls back gracefully if BrowserInput is unavailable.
    """
    specs: list[dict[str, Any]] = []

    # Browser tool with authoritative schema when available
    try:
        if BrowserInput:
            try:
                browser_schema = BrowserInput.model_json_schema()
            except Exception:
                browser_schema = {"type": "object", "properties": {"action": {"type": "object"}}}
        else:
            browser_schema = {"type": "object", "properties": {"action": {"type": "object"}}}

        specs.append({
            "type": "function",
            "function": {
                "name": "browser",
                "description": "Local Chromium browser tool for web research",
                "parameters": browser_schema,
            },
        })
    except Exception:
        pass

    # http_request schema (minimal, robust)
    specs.append({
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP request (JSON preferred).",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]},
                    "url": {"type": "string"},
                    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    "params": {"type": "object", "additionalProperties": True},
                    "data": {"type": "string"},
                    "json": {"type": "object"},
                    "timeout": {"type": "number"}
                },
                "required": ["method", "url"],
            },
        },
    })

    # file_read schema (local/remote)
    specs.append({
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read local or remote files (PDF/CSV/text).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "url": {"type": "string"},
                    "selector": {"type": "string"}
                }
            },
        },
    })

    # calculator schema
    specs.append({
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic expressions and unit conversions.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    })

    # python_repl schema
    specs.append({
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": "Execute short Python snippets for complex logic.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    })

    # final_answer schema (critical for stable conclusion)
    specs.append({
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit the final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    })

    return specs

def normalize_browser_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Fill required fields for known browser actions when missing (safe defaults)."""
    action = args.get("action", {})
    if not isinstance(action, dict):
        return {"action": {}}
    t = action.get("type")
    if t == "init_session":
        action.setdefault("session_name", "main-session")
        action.setdefault("description", "Web research session")
    if t == "close":
        action.setdefault("session_name", "main-session")
    # Remove automatic session_name injection - let the browser tool handle it
    return {"action": action}

# fallback allowlist for strands browser action types (snake_case)
ALLOWED_BROWSER_TYPES_FALLBACK = {
    "init_session","list_local_sessions","navigate","click","type","evaluate",
    "press_key","get_text","get_html","screenshot","refresh","back","forward",
    "new_tab","switch_tab","close_tab","list_tabs","get_cookies","set_cookies",
    "network_intercept","execute_cdp","close"
}

# Prefer authoritative actions from Strands models (discriminated union)
def get_browser_actions_from_models() -> list[str]:
    try:
        from typing import get_args
        from strands_tools.browser.models import BrowserInput
        action_union = BrowserInput.model_fields["action"].annotation
        actions: set[str] = set()
        for action_cls in get_args(action_union):
            try:
                type_ann = action_cls.model_fields["type"].annotation
                literal_vals = [v for v in get_args(type_ann) if isinstance(v, str)]
                for v in literal_vals:
                    actions.add(v)
            except Exception:
                continue
        return sorted(actions)
    except Exception:
        return []


class BrowserExecutor:
    """Lightweight executor that normalizes, validates, and executes browser actions.

    - Normalizes missing fields (e.g., session_name)
    - Enforces allowlist based on authoritative model schema when available
    - Gentle rewrite: "search" -> navigate(duckduckgo?q=...)
    - Returns structured error info on validation failures
    """

    def __init__(self, tool: Any):
        self._tool = tool
        actions = get_browser_actions_from_models()
        self._allowed: set[str] = set(actions) if actions else set(ALLOWED_BROWSER_TYPES_FALLBACK)

    @property
    def allowed(self) -> list[str]:
        return sorted(self._allowed)

    def _rewrite_if_needed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = args.get("action", {}) or {}
            a_type = str(action.get("type", "")).strip().lower()
            if a_type == "search":
                query = str(action.get("query", "")).strip()
                if query:
                    return {
                        "action": {
                            "type": "navigate",
                            "url": f"https://duckduckgo.com/?q={query}",
                        }
                    }
        except Exception:
            pass
        return args

    def _normalize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_browser_args(args)
        normalized = self._rewrite_if_needed(normalized)
        return normalized

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        safe_args = self._normalize(args)
        action = safe_args.get("action", {}) or {}
        a_type = action.get("type")
        if not a_type or a_type not in self._allowed:
            return {
                "error": f"invalid browser.action.type '{a_type}'. Use one of {self.allowed}",
                "allowed_actions": self.allowed,
            }
        try:
            result = self._tool(safe_args)
            if asyncio.iscoroutine(result):
                result = await result
            return result if isinstance(result, dict) else {"result": result}
        except ValidationError as ve:
            # Provide structured, actionable error info
            return {
                "error": "validation_error",
                "details": ve.errors(),
                "hint": "Ensure required fields are present and types match the schema",
                "allowed_actions": self.allowed,
            }
        except Exception as e:
            return {"error": f"tool error: {e}"}


def create_browser_agent():
    """Create a StrandsAgent with browser capabilities using environment configuration."""
    load_dotenv(find_dotenv())
    from transformers import AutoTokenizer
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if together_api_key:
        openai_kwargs = {"api_key": together_api_key, "base_url": "https://api.together.xyz/v1"}
        model_id = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    elif openai_api_key:
        openai_kwargs = {"api_key": openai_api_key}
        if os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
        model_id = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("API key required")
    
    rollout_engine = RolloutEngine(engine_name="openai", tokenizer=tokenizer, openai_kwargs=openai_kwargs)
    
    max_tokens = 1000
    temperature = 0.2
        
    model = RLLMModel(rollout_engine=rollout_engine, model_id=model_id, max_tokens=max_tokens, temperature=temperature)

    headless = os.getenv("BROWSER_HEADLESS", "true").lower() in ("1", "true", "yes")
    try:
        browser = LocalChromiumBrowser(headless=headless)
    except TypeError:
        browser = LocalChromiumBrowser()

    # Create enhanced executor and system prompt with dynamic action list
    executor = BrowserExecutor(browser.browser)
    enhanced_system_prompt = (
        SYSTEM_PROMPT
        + "\nAllowed Browser Action Types: "
        + ", ".join(executor.allowed)
        + "\nAvailable Utility Tools: http_request, file_read, calculator, python_repl"
    )
    agent = StrandsAgent(
        model=model,
        system_prompt=enhanced_system_prompt,
        tools=[browser.browser, http_request, file_read, calculator, python_repl],
    )
    
    return agent, browser, executor