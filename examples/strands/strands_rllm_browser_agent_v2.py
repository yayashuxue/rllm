import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple
import re

from dotenv import load_dotenv, find_dotenv
from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools.browser import LocalChromiumBrowser


SYSTEM_PROMPT = """
You are a lean web research agent using a single 'browser' tool. Minimize steps by cutting uncertainty.

Planning & State (in Thought line):
- First declare Schema=[field1, field2, ...] (compact, task-specific; agent-defined).
- State focus_dim=<time|geo|definition|number> and briefly note bucket plan (e.g., 2010..2012, site:.gov, keyword: inaugural, threshold:>50,000).

Tooling:
- Use the 'browser' tool. Only use action types listed in the injected tool manifest.
- NEVER invent other tools or action types.
- Prefer evaluate/get_text/get_html for targeted extraction. When using evaluate, return a JSON object matching your Schema.
- If CAPTCHA encountered, immediately switch to DuckDuckGo (https://duckduckgo.com/?q=...) or Bing and continue.

STRICT OUTPUT (2 lines only):
Thought: <Schema=[...]; focus_dim=...; brief reasoning (<80 tokens)>
Action: {"name":"browser","arguments":{"action":{"type":"<one of allowed actions>", ...}}}
OR
Action: {"name":"final_answer","arguments":{"answer":"..."}}
"""

USER_TEMPLATE = """
Question: {question}
History:
{history}
{schema_hint}
Now produce the next Thought and Action.
"""

ACTION_RE = re.compile("Action:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)
SCHEMA_RE = re.compile("Schema\s*:\s*(\[[^\]]*\])", re.IGNORECASE)

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
    else:
        # For all other actions, ensure a session is targeted unless the action
        # does not require one (e.g., listing local sessions)
        if t and t != "list_local_sessions":
            action.setdefault("session_name", "main-session")
    return {"action": action}

# fallback allowlist for strands browser action types (snake_case)
ALLOWED_BROWSER_TYPES_FALLBACK = {
    "init_session","list_local_sessions","navigate","click","type","evaluate",
    "press_key","get_text","get_html","screenshot","refresh","back","forward",
    "new_tab","switch_tab","close_tab","list_tabs","get_cookies","set_cookies",
    "network_intercept","execute_cdp","close"
}

# Prefer authoritative actions from Strands Pydantic models when available
def get_browser_actions_from_models() -> list[str]:
    try:
        from typing import get_args
        from strands_tools.browser.models import BrowserInput
        union_ann = BrowserInput.model_fields["action"].annotation
        actions: set[str] = set()
        for action_cls in get_args(union_ann):
            try:
                type_ann = action_cls.model_fields["type"].annotation
                for v in get_args(type_ann):
                    if isinstance(v, str):
                        actions.add(v)
            except Exception:
                continue
        return sorted(actions)
    except Exception:
        return []

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

async def main():
    load_dotenv(find_dotenv())
    from transformers import AutoTokenizer
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    openai_kwargs: Dict[str, Any] = {}
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        openai_kwargs["base_url"] = base_url
    if api_key:
        openai_kwargs["api_key"] = api_key
    rollout_engine = RolloutEngine(engine_name="openai", tokenizer=tokenizer, openai_kwargs=openai_kwargs)
    model = RLLMModel(rollout_engine=rollout_engine, model_id=os.getenv("MODEL_NAME", "gpt-4o"), max_tokens=350, temperature=0.2)

    headless = os.getenv("BROWSER_HEADLESS", "true").lower() in ("1", "true", "yes")
    try:
        browser = LocalChromiumBrowser(headless=headless)
    except TypeError:
        browser = LocalChromiumBrowser()

    agent = StrandsAgent(model=model, system_prompt=SYSTEM_PROMPT, tools=[browser.browser])
    # Display authoritative action list from Strands models (for user visibility only)
    try:
        print("Authoritative browser actions:", get_browser_actions_from_models())
    except Exception:
        pass

    question = os.getenv(
        "QUESTION",
        # "There was an early Christian poetic hymn composed by a late antique writer who passed away around the mid-5th century. The year of this writer’s death coincides with the last year of a scientific chronology that reconstructs environmental conditions from several centuries before the modern era. What is the name of this chronology?",
        # "Ap musical piece closely associated with a prominent South American capital features lyrics written by a notable figure who was later recognized with a distinguished local civic honor in the early 21st century. The composition’s melody was created by a musician who received formal training at a respected arts institution in western Colombia. What is the name of this musical piece?",
    )

    print("=== Strands Browser Research (minimal) ===")
    print(f"Model: {agent.model.get_config().get('model_id')}")
    print("Tools: ['browser']")
    print("Question:")
    print(question)

    try:
        # Initialize a browser session explicitly to encourage stable behavior
        init_payload = {
            "action": {
                "type": "init_session",
                "session_name": "main-session",
                "description": "Web research session"
            }
        }
        init_result = browser.browser(init_payload)
        if asyncio.iscoroutine(init_result):
            init_result = await init_result
        print("\n--- Browser init ---")
        print(json.dumps(init_result, ensure_ascii=False)[:400])

        # Minimal loop: use StrandsAgent for planning, we execute browser actions
        history: list[str] = []
        current_schema: Optional[list] = None
        max_steps = int(os.getenv("MAX_STEPS", "30"))
        for step in range(1, max_steps + 1):
            print(f"[step {step}] Building prompt", flush=True)
            schema_hint = f"Current Schema: {json.dumps(current_schema)}" if current_schema else ""
            user = USER_TEMPLATE.format(question=question, history="".join(history), schema_hint=schema_hint)
            print(f"[step {step}] Querying model {agent.model.get_config().get('model_id')}...", flush=True)
            resp = await agent.invoke_async(user)
            text = str(resp)
            preview = text.replace("\n", " ")[:200]
            print(f"[step {step}] LLM output: {preview}{'...' if len(text) > 200 else ''}", flush=True)
            action, thought = parse_action(text)
            history.append(f"Thought: {thought or '(missing)'}")
            try:
                sch = parse_schema(thought)
                if sch:
                    current_schema = sch
            except Exception:
                pass
            if not action or "name" not in action:
                print(f"[step {step}] Parse error: no valid Action JSON found", flush=True)
                history.append("Observation: (parse_error) Output exactly two lines: Thought + Action JSON (browser/final_answer)")
                continue
            name = str(action.get("name", "")).strip()
            args = action.get("arguments", {}) or {}
            try:
                args_preview = json.dumps(args)[:200]
            except Exception:
                args_preview = str(args)[:200]
            print(f"[step {step}] Dispatching action: {name} args={args_preview}", flush=True)
            if name == "final_answer":
                answer = str(args.get("answer", "")).strip()
                print(f"[step {step}] Final answer received", flush=True)
                print("\n=== Final Answer ===")
                print(answer)
                return
            obs: Dict[str, Any]
            try:
                if name == "browser":
                    safe_args = normalize_browser_args(args)
                    a = safe_args.get("action", {})
                    a_type = a.get("type")
                    # dynamically get allowed actions from model/tool manifest
                    allowed = set()
                    try:
                        actions = get_browser_actions_from_models()
                        allowed = set(actions) if actions else set(ALLOWED_BROWSER_TYPES_FALLBACK)
                    except Exception:
                        allowed = set(ALLOWED_BROWSER_TYPES_FALLBACK)
                    if not allowed:
                        allowed = set(ALLOWED_BROWSER_TYPES_FALLBACK)
                    if a_type not in allowed:
                        obs = {"error": f"invalid browser.action.type '{a_type}'. Use one of {sorted(allowed)}"}
                    else:
                        result = browser.browser(safe_args)
                        if asyncio.iscoroutine(result):
                            result = await result
                        obs = result if isinstance(result, dict) else {"result": result}
                else:
                    obs = {"error": f"unknown action: {name}. Only 'browser' or 'final_answer' are allowed."}
            except Exception as e:
                obs = {"error": f"tool error: {e}"}
            obs_text = json.dumps(obs, ensure_ascii=False)
            print(f"[step {step}] Observation: {obs_text[:1600]}", flush=True)
            history.append(f"Observation: {obs_text[:1600]}")
        print("(FAILED: step limit)")
    except Exception as e:
        print(f"Error running agent: {e}")


if __name__ == "__main__":
    asyncio.run(main())
