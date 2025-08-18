import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple
import re

from dotenv import load_dotenv, find_dotenv
from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools.browser import LocalChromiumBrowser
from pydantic import ValidationError
try:
    from strands_tools.browser.models import BrowserInput
except ImportError:
    BrowserInput = None


SYSTEM_PROMPT = """
You are a web research agent using a 'browser' tool. Follow the MANDATORY workflow to avoid loops.

MANDATORY WORKFLOW (must follow in order):
1. navigate to search page → get_text to read results → navigate to promising result → get_text to extract content
2. If no useful content found: try different search terms or sources
3. Once you have sufficient info: final_answer

Planning & State (in Thought line):
- First declare Schema=[field1, field2, ...] (compact, task-specific; agent-defined).
- State focus_dim=<time|geo|definition|number> and brief reasoning (<80 tokens).
- Always mention your current workflow step (navigate/get_text/analyze/conclude).

Tooling:
- Use ONLY the 'browser' tool with actions from the manifest.
- NEVER repeat the same navigate action - always follow with get_text!
- To search: navigate to "https://duckduckgo.com/html/?q==your+search+terms"
- After navigate: MUST use get_text with selector (required parameter)
- Safe selectors: "body" (full page), "main", "[data-testid]", "h1,h2,h3", "p"
- Example: {"type":"get_text","selector":"body"} - reads full page text
- If selector times out, try simpler ones: "body" > "main" > "p" > get_html
- For extraction: use evaluate with JSON matching your Schema
- If stuck in loops: change search terms or conclude with available info

STRICT OUTPUT (2 lines only):
Thought: <Schema=[...]; workflow_step=navigate/get_text/conclude; reasoning>
Action: {"name":"browser","arguments":{"action":{"type":"<action>", ...}}}
OR
Action: {"name":"final_answer","arguments":{"answer":"..."}}
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


def normalize_browser_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Fill required fields for known browser actions when missing (safe defaults)."""
    action = args.get("action", {})
    if not isinstance(action, dict):
        return {"action": {}}
    t = action.get("type")
    if t == "init_session":
        action.setdefault("session_name", "main-session")
        action.setdefault("description", "Web research session")
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
    enhanced_system_prompt = SYSTEM_PROMPT + "\nAllowed Browser Action Types: " + ", ".join(executor.allowed)
    agent = StrandsAgent(model=model, system_prompt=enhanced_system_prompt, tools=[browser.browser])
    
    return agent, browser, executor