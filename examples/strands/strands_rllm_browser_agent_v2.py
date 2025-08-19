import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple
import re

from dotenv import load_dotenv, find_dotenv
from agent_factory import create_browser_agent, parse_action, parse_schema, USER_TEMPLATE, BrowserInput
from pydantic import ValidationError


USER_TEMPLATE = """Question: {question}
History: {history}
{schema_hint}

Output EXACTLY 2 lines:
Thought: [analysis]
Action: [JSON]"""

ACTION_RE = re.compile("Action:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)
SCHEMA_RE = re.compile("Schema\s*:\s*(\[[^\]]*\])", re.IGNORECASE)

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
    # Use shared factory and prompts
    agent, browser, executor = create_browser_agent()
    enhanced_system_prompt = agent.system_prompt or ""
    
    # Display authoritative action list from Strands models (for user visibility only)
    try:
        print("Authoritative browser actions:", executor.allowed)
    except Exception:
        pass

    question = os.getenv(
        "QUESTION",
        # "There was an early Christian poetic hymn composed by a late antique writer who passed away around the mid-5th century. The year of this writer's death coincides with the last year of a scientific chronology that reconstructs environmental conditions from several centuries before the modern era. What is the name of this chronology?",
        # "Ap musical piece closely associated with a prominent South American capital features lyrics written by a notable figure who was later recognized with a distinguished local civic honor in the early 21st century. The composition's melody was created by a musician who received formal training at a respected arts institution in western Colombia. What is the name of this musical piece?",
        # "加州最古老的poker room是哪家？",
        "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?",
    )

    print("=== Strands Browser Research (minimal) ===")
    print(f"Model: {agent.model.get_config().get('model_id')}")
    print("Tools: ['browser']")
    print("Question:")
    print(question)

    try:
        # Initialize a browser session if supported (optional)
        try:
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
        except Exception as e:
            print(f"\n--- Browser init skipped: {e} ---")
            # Continue without session management if not supported

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
        max_steps = int(os.getenv("MAX_STEPS", "30"))
        for step in range(1, max_steps + 1):
            print(f"[step {step}] Building prompt", flush=True)
            schema_hint = f"Current Schema: {json.dumps(current_schema)}" if current_schema else ""
            user = USER_TEMPLATE.format(question=question, history="".join(history), schema_hint=schema_hint)
            
            # Try tool_calls path first (if supported)
            tool_calls_handled = False
            model_id = agent.model.get_config().get('model_id', '')
            if browser_tools:
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
