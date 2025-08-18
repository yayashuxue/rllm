"""
Reusable GAIA evaluation script for Strands agents.

This script provides a standalone GAIA evaluation that can be used by any Strands example.
It handles GAIA dataset loading, answer validation, and result reporting.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TestDataset
from rllm.integrations.strands import StrandsAgent
from agent_factory import create_browser_agent, parse_action, parse_schema, USER_TEMPLATE, BrowserInput


class GAIAWorkflow:
    """Workflow for GAIA evaluation with Strands agents."""
    
    def __init__(self, agent: StrandsAgent, browser=None, executor=None):
        self.agent = agent
        self.browser = browser
        self.executor = executor
        
    async def run_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single GAIA task and return results."""
        # Handle both formats: direct load vs preprocessed
        question = task.get("Question") or task.get("problem", "")
        ground_truth = task.get("Final answer") or task.get("tests", "")
        task_id = task.get("task_id", "unknown")
        
        print(f"\n=== GAIA Task {task_id} ===")
        print(f"Question: {question}")
        print(f"📝 Ground Truth: {ground_truth}")
        print("-" * 50)
        
        try:
            # Initialize browser session if available (optional)
            if self.browser:
                try:
                    init_payload = {
                        "action": {
                            "type": "init_session",
                            "session_name": "main-session",
                            "description": f"GAIA task {task_id}"
                        }
                    }
                    init_result = self.browser.browser(init_payload)
                    if asyncio.iscoroutine(init_result):
                        await init_result
                except Exception:
                    # Browser might not need session management - continue without it
                    pass
            
            # Reset agent trajectory for new task
            self.agent.reset_trajectory(task=task)
            
            # Run the interactive task workflow (like v2)
            final_answer = await self._run_interactive_task(question)
            
            # If no answer found, try simple invoke as fallback
            if not final_answer:
                result = await self.agent.invoke_async(question)
                final_answer = self._extract_final_answer(result)
            
            # Validate answer
            is_correct = self._validate_answer(final_answer, ground_truth)
            
            return {
                "task_id": task_id,
                "question": question,
                "predicted_answer": final_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "trajectory": self.agent.trajectory,
                "num_steps": len(self.agent.trajectory.steps)
            }
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            return {
                "task_id": task_id,
                "question": question,
                "predicted_answer": "",
                "ground_truth": ground_truth,
                "is_correct": False,
                "error": str(e),
                "trajectory": None,
                "num_steps": 0
            }
        finally:
            # Clean up browser session if available
            if self.browser:
                try:
                    close_action = {"action": {"type": "close"}}
                    self.browser.browser(close_action)
                except Exception as cleanup_error:
                    print(f"Browser cleanup error: {cleanup_error}")
    
    def _extract_final_answer(self, result: Any) -> str:
        """Extract final answer from agent result."""
        if hasattr(result, 'message') and result.message:
            if hasattr(result.message, 'content'):
                if isinstance(result.message.content, list):
                    # Extract text from content blocks
                    text_content = ""
                    for block in result.message.content:
                        if isinstance(block, dict) and 'text' in block:
                            text_content += block['text']
                    return text_content.strip()
                else:
                    return str(result.message.content).strip()
        return str(result).strip() if result else ""
    
    def _validate_answer(self, predicted: str, ground_truth: str) -> bool:
        """Validate predicted answer against ground truth."""
        if not predicted or not ground_truth:
            return False
        
        # Simple string matching for now
        # TODO: Implement more sophisticated answer validation
        predicted_clean = predicted.lower().strip()
        ground_truth_clean = ground_truth.lower().strip()
        
        return predicted_clean == ground_truth_clean
    
    async def _run_interactive_task(self, question: str, max_steps: int = 15) -> str:
        """Run the interactive browser task workflow from v2."""
        if not self.executor:
            return ""
            
        try:
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
                except Exception:
                    pass
            
            # Interactive loop from v2
            history = []
            current_schema = None
            
            for step in range(1, max_steps + 1):
                schema_hint = f"Current Schema: {json.dumps(current_schema)}" if current_schema else ""
                user = USER_TEMPLATE.format(question=question, history="".join(history), schema_hint=schema_hint)
                
                # Try tool_calls path first (if supported)
                tool_calls_handled = False
                if browser_tools:
                    try:
                        enhanced_system_prompt = self.agent.system_prompt
                        chat_messages = [
                            {"role": "system", "content": enhanced_system_prompt},
                            {"role": "user", "content": user},
                        ]
                        resp_msg = await self.agent.model.rollout_engine.get_model_response(
                            chat_messages,
                            model=self.agent.model.get_config().get("model_id"),
                            tools=browser_tools,
                            tool_choice="auto",
                            return_message_dict=True,
                            **self.agent.model.get_config().get("params", {}),
                        )
                        
                        tool_calls = resp_msg.get("tool_calls") if isinstance(resp_msg, dict) else None
                        if tool_calls:
                            self.agent._start_new_step(observation=user)
                            model_response_content = resp_msg.get("content", "")
                            
                            for tc in tool_calls:
                                tc_name = tc.get("name")
                                tc_args_str = tc.get("arguments", "")
                                
                                if tc_name == "browser":
                                    try:
                                        tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                                        obs = await self.executor.execute(tc_args)
                                    except Exception as e:
                                        obs = {"error": f"tool_call error: {e}"}
                                    
                                    obs_text = json.dumps(obs, ensure_ascii=False)
                                    history.append(f"Observation: {obs_text[:1600]}")
                                    tool_calls_handled = True
                                    
                                    self.agent._finish_current_step(
                                        model_response=model_response_content + f" [tool_call:{tc_name}]",
                                        action={"tool_call": tc_name, "arguments": tc_args},
                                        done=False
                                    )
                                    
                                elif tc_name == "final_answer":
                                    try:
                                        tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                                        answer = str(tc_args.get("answer", "")).strip()
                                        
                                        self.agent._finish_current_step(
                                            model_response=model_response_content + f" [tool_call:{tc_name}]",
                                            action={"tool_call": tc_name, "arguments": tc_args},
                                            done=True
                                        )
                                        return answer
                                    except Exception:
                                        pass
                            
                            if tool_calls_handled:
                                continue
                    except Exception:
                        pass
                
                # Fallback to text parsing
                resp = await self.agent.invoke_async(user)
                text = str(resp) if resp is not None else ""
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
                    history.append("Observation: (parse_error) Output exactly two lines: Thought + Action JSON (browser/final_answer)")
                    continue
                    
                name = str(action.get("name", "")).strip()
                args = action.get("arguments", {}) or {}
                
                if name == "final_answer":
                    answer = str(args.get("answer", "")).strip()
                    return answer
                    
                try:
                    if name == "browser":
                        obs = await self.executor.execute(args)
                    else:
                        obs = {"error": f"unknown action: {name}. Only 'browser' or 'final_answer' are allowed."}
                except Exception as e:
                    obs = {"error": f"tool error: {e}"}
                    
                obs_text = json.dumps(obs, ensure_ascii=False)
                history.append(f"Observation: {obs_text[:1600]}")
                
            return ""  # No final answer found
            
        except Exception as e:
            print(f"Interactive task error: {e}")
            return ""


def load_gaia_dataset(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load GAIA dataset from the downloaded JSON file."""
    try:
        # Try rLLM's standardized data loading first
        try:
            data = load_dataset(TestDataset.Web.GAIA)
            if isinstance(data, list):
                dataset = data
            else:
                dataset = list(data)
        except Exception:
            # Fallback to direct file loading
            import rllm
            rllm_path = os.path.dirname(os.path.dirname(rllm.__file__))
            data_path = os.path.join(rllm_path, "rllm/data/train/web/gaia.json")
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"GAIA data not found at {data_path}")
            
            with open(data_path, 'r') as f:
                dataset = json.load(f)
        
        # Apply limit if specified
        if limit:
            dataset = dataset[:limit]
        
        print(f"Loaded {len(dataset)} GAIA tasks")
        return dataset
        
    except Exception as e:
        print(f"Error loading GAIA dataset: {e}")
        print("Make sure GAIA data is downloaded. Run: python scripts/data/download_gaia.py")
        return []


async def evaluate_gaia(
    agent: StrandsAgent,
    browser=None,
    executor=None,
    limit: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a Strands agent on GAIA dataset.
    
    Args:
        agent: StrandsAgent instance to evaluate
        limit: Maximum number of tasks to evaluate (None for all)
        output_file: Path to save detailed results (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    print("=== GAIA Evaluation for Strands Agent ===")
    
    # Load dataset
    dataset = load_gaia_dataset(limit=limit)
    if not dataset:
        return {"error": "Failed to load GAIA dataset"}
    
    # Create workflow
    workflow = GAIAWorkflow(agent, browser, executor)
    
    # Run evaluation
    results = []
    correct = 0
    total = len(dataset)
    
    for i, task in enumerate(dataset):
        print(f"\nProgress: {i+1}/{total}")
        result = await workflow.run_single_task(task)
        results.append(result)
        
        if result.get("is_correct", False):
            correct += 1
            print(f"✅ Correct ({correct}/{i+1})")
            print(f"   🎯 Answer: {result.get('predicted_answer', '')}")
        else:
            print(f"❌ Incorrect ({correct}/{i+1})")
            print(f"   🤖 Predicted: {result.get('predicted_answer', '')}")
            print(f"   ✅ Expected:  {result.get('ground_truth', '')}")
            print()
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    evaluation_results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "model": str(agent.model.get_config().get("model_id", "unknown")),
        "detailed_results": results
    }
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\n=== GAIA Evaluation Summary ===")
    print(f"Model: {evaluation_results['model']}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Average steps per task: {sum(r.get('num_steps', 0) for r in results) / len(results):.1f}")
    
    return evaluation_results


def create_gaia_agent():
    """Create a StrandsAgent for GAIA evaluation using environment configuration."""
    return create_browser_agent()


async def main():
    """Main evaluation script."""
    # Get evaluation parameters
    limit = int(os.getenv("GAIA_LIMIT", "10"))  # Default to 10 for testing
    output_file = os.getenv("GAIA_OUTPUT", "gaia_evaluation_results.json")
    
    print(f"Evaluating on {limit} GAIA tasks")
    try:
        # Create agent with browser capabilities
        agent, browser, executor = create_gaia_agent()
        
        print(f"Using model: {agent.model.get_config().get('model_id')}")
        print(f"Browser actions: {len(executor.allowed)} available")
        
        # Run evaluation
        results = await evaluate_gaia(
            agent=agent,
            browser=browser,
            executor=executor,
            limit=limit,
            output_file=output_file
        )
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())