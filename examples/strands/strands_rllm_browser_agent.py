import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple
import re

from dotenv import load_dotenv, find_dotenv
from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from strands_tools.browser import LocalChromiumBrowser


SYSTEM_PROMPT = (
    "You are a lean web research agent using a browser tool. Minimize steps by cutting uncertainty.\n"
    "- Decide buckets (time ranges, site:, numeric thresholds, role keywords) and navigate accordingly.\n"
    "- Extract only: city, year, organizer_name, organizer_last_initial, attendance, source_url.\n"
    "- Use the 'browser' tool with actions: init_session, list_local_sessions, navigate, click, type, evaluate, press_key, get_text, get_html, screenshot, refresh, back, forward, new_tab, switch_tab, close_tab, list_tabs, get_cookies, set_cookies, network_intercept, execute_cdp, close.\n"
    "- Common required fields: init_session{session_name(^[a-z0-9-]+$), description}; navigate{url}; click{selector}; type{selector, text}; get_text{selector}.\n"
    "- Prefer get_text/get_html for extraction; use navigate with explicit URLs.\n"
    "- If CAPTCHA encountered, immediately switch to DuckDuckGo (https://duckduckgo.com/?q=...) or Bing and continue.\n\n"
    "STRICT OUTPUT (2 lines only):\n"
    "Thought: <concise reasoning (<50 tokens)>\n"
    "Action: {\"name\":\"browser\",\"arguments\":{\"action\":{\"type\":\"<one of allowed actions>\", ...}}}\n"
    "OR\n"
    "Action: {\"name\":\"final_answer\",\"arguments\":{\"answer\":\"...\"}}\n"
)

USER_TEMPLATE = (
    "Question: {question}\n"
    "History:\n{history}\n"
    "Now produce the next Thought and Action.\n"
)

ACTION_RE = re.compile(r"Action:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)


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


def make_normalize_fn(session_ref: Dict[str, str]):
    def _normalize(args: Dict[str, Any]) -> Dict[str, Any]:
        action = args.get("action", {})
        if not isinstance(action, dict):
            return {"action": {}}
        t = action.get("type")
        if t == "init_session":
            # Ensure valid defaults and pattern-safe session name
            name = action.get("session_name") or session_ref.get("name") or "main-session"
            name = str(name).lower().replace("_", "-")
            action["session_name"] = name
            action.setdefault("description", "Web research session")
        else:
            # Most actions require session_name; inject last known
            if "session_name" not in action:
                action["session_name"] = session_ref.get("name") or "main-session"
        return {"action": action}
    return _normalize

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

    question = os.getenv(
        "QUESTION",
        # "There was an early Christian poetic hymn composed by a late antique writer who passed away around the mid-5th century. The year of this writer’s death coincides with the last year of a scientific chronology that reconstructs environmental conditions from several centuries before the modern era. What is the name of this chronology?",
        # "A musical piece closely associated with a prominent South American capital features lyrics written by a notable figure who was later recognized with a distinguished local civic honor in the early 21st century. The composition’s melody was created by a musician who received formal training at a respected arts institution in western Colombia. What is the name of this musical piece?",
    )

    print("=== Strands Browser Research (minimal) ===")
    print(f"Model: {agent.model.get_config().get('model_id')}")
    print("Tools: ['browser']")
    print("Question:")
    print(question)

    try:

        # Minimal loop: use StrandsAgent for planning, we execute browser actions
        history: list[str] = []
        max_steps = int(os.getenv("MAX_STEPS", "30"))
        session_state: Dict[str, str] = {"name": "main-session"}
        normalize_browser_args = make_normalize_fn(session_state)
        for step in range(1, max_steps + 1):
            print(f"[step {step}] Building prompt", flush=True)
            user = USER_TEMPLATE.format(question=question, history="\n".join(history))
            print(f"[step {step}] Querying model {agent.model.get_config().get('model_id')}...", flush=True)
            resp = await agent.invoke_async(user)
            text = str(resp)
            preview = text.replace("\n", " ")[:200]
            print(f"[step {step}] LLM output: {preview}{'...' if len(text) > 200 else ''}", flush=True)
            action, thought = parse_action(text)
            history.append(f"Thought: {thought or '(missing)'}")
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
                    result = browser.browser(safe_args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    # Track session name once initialized
                    try:
                        act = safe_args.get("action", {})
                        if act.get("type") == "init_session":
                            sn = act.get("session_name")
                            if isinstance(sn, str) and sn:
                                session_state["name"] = sn
                    except Exception:
                        pass
                    obs = result if isinstance(result, dict) else {"result": result}
                else:
                    obs = {"error": f"unknown action: {name}"}
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
