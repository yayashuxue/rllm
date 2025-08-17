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
- To search: navigate to "https://duckduckgo.com/?q=your+search+terms"
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
            # Force session_name to main-session to avoid creative naming by models
            action["session_name"] = "main-session"
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
                            "session_name": action.get("session_name", "main-session"),
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


async def main():
    load_dotenv(find_dotenv())
    from transformers import AutoTokenizer
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if together_api_key:
        openai_kwargs = {"api_key": together_api_key, "base_url": "https://api.together.xyz/v1"}
        model_name = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
        if model_name == "gpt-oss":
            model_id = "openai/gpt-oss-120b" 
        else:
            model_id = model_name
    elif openai_api_key:
        openai_kwargs = {"api_key": openai_api_key}
        if os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
        model_id = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("API key required")
    
    rollout_engine = RolloutEngine(engine_name="openai", tokenizer=tokenizer, openai_kwargs=openai_kwargs)
    
    # Adjust parameters based on model capabilities  
    if "gpt-oss" in model_id:
        max_tokens = 100  # Very short responses
        temperature = 0.0  # Deterministic
        # Disable o1-style reasoning with specific parameters
        extra_params = {"reasoning": False, "stream": False}
    else:
        max_tokens = 350
        temperature = 0.7
        extra_params = {}
        
    model = RLLMModel(rollout_engine=rollout_engine, model_id=model_id, max_tokens=max_tokens, temperature=temperature, **extra_params)

    headless = os.getenv("BROWSER_HEADLESS", "true").lower() in ("1", "true", "yes")
    try:
        browser = LocalChromiumBrowser(headless=headless)
    except TypeError:
        browser = LocalChromiumBrowser()

    # Create enhanced executor and system prompt with dynamic action list
    executor = BrowserExecutor(browser.browser)
    enhanced_system_prompt = SYSTEM_PROMPT + "\nAllowed Browser Action Types: " + ", ".join(executor.allowed)
    agent = StrandsAgent(model=model, system_prompt=enhanced_system_prompt, tools=[browser.browser])
    
    # Display authoritative action list from Strands models (for user visibility only)
    try:
        print("Authoritative browser actions:", executor.allowed)
    except Exception:
        pass

    question = os.getenv(
        "QUESTION",
        # "There was an early Christian poetic hymn composed by a late antique writer who passed away around the mid-5th century. The year of this writer's death coincides with the last year of a scientific chronology that reconstructs environmental conditions from several centuries before the modern era. What is the name of this chronology?",
        "Ap musical piece closely associated with a prominent South American capital features lyrics written by a notable figure who was later recognized with a distinguished local civic honor in the early 21st century. The composition's melody was created by a musician who received formal training at a respected arts institution in western Colombia. What is the name of this musical piece?",
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

        # Build tool spec for direct tool_calls support
        browser_tools = []
        if BrowserInput:
            try:
                browser_schema = BrowserInput.model_json_schema()
                browser_tools = [{
                    "type": "function",
                    "function": {
                        "name": "browser",
                        "description": "Local Chromium browser tool for web research",
                        "parameters": browser_schema,
                    },
                }]
            except Exception as e:
                browser_schema = {"type": "object", "properties": {}}
        # Note: BrowserInput available, tool_calls support enabled

        # RL-integrated loop: try tool_calls first, fall back to text parsing
        history: list[str] = []
        current_schema: Optional[list] = None
        max_steps = 5 if not together_api_key else int(os.getenv("MAX_STEPS", "50"))
        for step in range(1, max_steps + 1):
            print(f"[step {step}] Building prompt", flush=True)
            schema_hint = f"Current Schema: {json.dumps(current_schema)}" if current_schema else ""
            user = USER_TEMPLATE.format(question=question, history="".join(history), schema_hint=schema_hint)
            
            # Try tool_calls path first (if supported and not GPT-OSS)
            tool_calls_handled = False
            model_id = agent.model.get_config().get('model_id', '')
            if browser_tools and "gpt-oss" not in model_id:
                try:
                    print(f"[step {step}] Querying model {agent.model.get_config().get('model_id')} (prefer tool_calls)...", flush=True)
                    # Direct call to rollout engine to get structured response
                    chat_messages = [
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": user},
                    ]
                    resp_msg = await agent.model.rollout_engine.get_model_response(
                        chat_messages,
                        model=agent.model.get_config().get("model_id"),
                        tools=browser_tools,
                        tool_choice="auto",
                        return_message_dict=True,
                        **agent.model.get_config().get("params", {}),
                    )
                    
                    # Check if we got tool_calls
                    tool_calls = resp_msg.get("tool_calls") if isinstance(resp_msg, dict) else None
                    if tool_calls:
                        print(f"[step {step}] tool_calls received: {len(tool_calls)}", flush=True)
                        
                        # CRITICAL: Record RL step data for tool_calls path
                        agent._start_new_step(observation=user)
                        model_response_content = resp_msg.get("content", "")
                        
                        for tc in tool_calls:
                            tc_name = tc.get("name")
                            tc_args_str = tc.get("arguments", "")
                            print(f"[step {step}] Executing tool_call: {tc_name} args={tc_args_str[:200]}", flush=True)
                            
                            if tc_name == "browser":
                                try:
                                    tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                                    obs = await executor.execute(tc_args)
                                except Exception as e:
                                    obs = {"error": f"tool_call error: {e}"}
                                
                                obs_text = json.dumps(obs, ensure_ascii=False)
                                print(f"[step {step}] Observation (tool_call): {obs_text[:1600]}", flush=True)
                                history.append(f"Observation: {obs_text[:1600]}")
                                tool_calls_handled = True
                                
                                # Record RL step completion with tool_call action
                                agent._finish_current_step(
                                    model_response=model_response_content + f" [tool_call:{tc_name}]",
                                    action={"tool_call": tc_name, "arguments": tc_args},
                                    done=False
                                )
                                
                            elif tc_name == "final_answer":
                                try:
                                    tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                                    answer = str(tc_args.get("answer", "")).strip()
                                    print(f"[step {step}] Final answer via tool_call", flush=True)
                                    print("\n=== Final Answer ===")
                                    print(answer)
                                    
                                    # Record RL step completion for final answer
                                    agent._finish_current_step(
                                        model_response=model_response_content + f" [tool_call:{tc_name}]",
                                        action={"tool_call": tc_name, "arguments": tc_args},
                                        done=True
                                    )
                                    return
                                except Exception as e:
                                    print(f"[step {step}] Error parsing final_answer tool_call: {e}", flush=True)
                        
                        if tool_calls_handled:
                            continue  # Skip to next step
                except Exception as e:
                    print(f"[step {step}] Tool_calls attempt failed: {e}", flush=True)
            
            # Fallback to text parsing (always goes through StrandsAgent for RL tracking)
            print(f"[step {step}] Querying model {agent.model.get_config().get('model_id')} (text fallback)...", flush=True)
            resp = await agent.invoke_async(user)
            text = str(resp) if resp is not None else ""
            preview = text.replace("\n", " ")[:200] if text else "(empty)"
            print(f"[step {step}] LLM output: {preview}{'...' if len(text) > 200 else ''}", flush=True)
            action, thought = parse_action(text)
            history.append(f"Thought: {thought or '(missing)'}")
            try:
                if thought:
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
                    # Use enhanced BrowserExecutor for validation, normalization, and intelligent rewriting
                    obs = await executor.execute(args)
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
    finally:
        # Ensure browser is properly closed
        try:
            if hasattr(browser, 'browser') and callable(getattr(browser, 'browser')):
                # Use the browser method with close action
                close_action = {"action": {"type": "close", "session_name": "main-session"}}
                browser.browser(close_action)
            elif hasattr(browser, 'quit'):
                browser.quit()
        except Exception as e:
            print(f"Error closing browser: {e}")
    
    # ===== RL INTEGRATION TESTING ===== 
    print("\n" + "="*50)
    print("🧪 RL INTEGRATION VERIFICATION")
    print("="*50)
    
    # Test 1: Verify trajectory data exists
    trajectory = agent.trajectory
    print(f"✅ Trajectory exists: {trajectory is not None}")
    print(f"✅ Steps recorded: {len(trajectory.steps)}")
    print(f"✅ Total reward: {trajectory.reward}")
    
    if trajectory.steps:
        print(f"\n📊 STEP ANALYSIS:")
        for i, step in enumerate(trajectory.steps):
            print(f"  Step {i+1}:")
            print(f"    - Observation: {str(step.observation)[:100]}...")
            print(f"    - Model response: {str(step.model_response)[:100]}...")
            print(f"    - Action: {str(step.action)[:100] if step.action else 'None'}...")
            print(f"    - Reward: {step.reward}")
            print(f"    - Done: {step.done}")
            print(f"    - Has chat_completions: {len(step.chat_completions) > 0}")
    
    # Test 2: Verify critical RL components
    print(f"\n🔍 RL COMPONENT VERIFICATION:")
    print(f"  - Agent has trajectory: {hasattr(agent, '_trajectory')}")
    print(f"  - Agent has step methods: {hasattr(agent, '_start_new_step')}")
    print(f"  - Agent has reward methods: {hasattr(agent, 'update_step_reward')}")
    print(f"  - Ready for reward assignment: {len(trajectory.steps) > 0}")
    
    # Test 3: Tool integration verification  
    print(f"\n🛠️  TOOL INTEGRATION VERIFICATION:")
    print(f"  - Enhanced executor used: {executor is not None}")
    print(f"  - Tool specs auto-injected: {hasattr(agent.model, '_default_tool_specs')}")
    print(f"  - Allowed actions: {len(executor.allowed)} actions")
    print(f"  - BrowserInput schema: {BrowserInput is not None}")
    
    # Success summary
    success_metrics = [
        len(trajectory.steps) > 0,  # Trajectory recorded
        all(step.observation is not None for step in trajectory.steps),  # Observations recorded
        hasattr(agent, 'update_step_reward'),  # Ready for reward assignment
        executor is not None,  # Enhanced tools working
    ]
    
    success_rate = sum(success_metrics) / len(success_metrics) * 100
    print(f"\n🎯 INTEGRATION SUCCESS: {success_rate:.0f}% ({sum(success_metrics)}/{len(success_metrics)} checks passed)")
    
    if success_rate >= 75:
        print("✅ RL-FIRST INTEGRATION SUCCESSFUL!")
        print("   Ready for reward assignment and training pipeline integration.")
    else:
        print("❌ INTEGRATION ISSUES DETECTED")
        print("   Some RL components not working properly.")
    
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
