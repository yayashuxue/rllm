import asyncio
import numpy as np
from collections import defaultdict

from countdown_agent import CountDownAgent
from countdown_reward import countdown_reward_fn
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.utils import compute_pass_at_k


def count_tokens(text, tokenizer):
    """Count tokens in a text string."""
    return len(tokenizer.encode(text))


def analyze_trajectory_statistics(results, tokenizer, max_context_length=4096):
    """
    Analyze trajectory statistics at different context length cutoffs.
    
    Args:
        results: List of trajectory results
        tokenizer: Tokenizer to count tokens
        max_context_length: Maximum context window size
    """
    print("\n" + "="*80)
    print("TRAJECTORY STATISTICS ANALYSIS")
    print("="*80)
    
    # Debug: Print structure of first result
    if results:
        print(f"Debug: First result type: {type(results[0])}")
        if hasattr(results[0], '__dict__'):
            print(f"Debug: First result attributes: {list(results[0].__dict__.keys())}")
        if hasattr(results[0], 'steps') and results[0].steps:
            print(f"Debug: First step type: {type(results[0].steps[0])}")
            if hasattr(results[0].steps[0], '__dict__'):
                print(f"Debug: First step attributes: {list(results[0].steps[0].__dict__.keys())}")
    print()
    
    # Extract trajectory data
    trajectory_data = []
    for i, result in enumerate(results):
        # Handle different result structures
        if hasattr(result, 'trajectory'):
            trajectory = result.trajectory
        else:
            # result is likely a Trajectory object itself
            trajectory = result
        
        if not hasattr(trajectory, 'steps') or not trajectory.steps:
            continue
            
        # Get the final step
        final_step = trajectory.steps[-1]
        
        # Count tokens in the conversation
        conversation_text = ""
        if hasattr(final_step, 'chat_completions') and final_step.chat_completions:
            for msg in final_step.chat_completions:
                conversation_text += msg.get('content', '') + " "
        elif hasattr(final_step, 'model_response') and hasattr(final_step, 'observation'):
            conversation_text = str(getattr(final_step, 'observation', '')) + " " + str(getattr(final_step, 'model_response', ''))
        elif hasattr(final_step, 'observation'):
            conversation_text = str(getattr(final_step, 'observation', ''))
        else:
            # Try to get any text content from the step
            conversation_text = str(final_step)
            
        token_count = count_tokens(conversation_text, tokenizer)
        
        # Determine if answer was correct
        is_correct = False
        if hasattr(final_step, 'reward') and final_step.reward is not None:
            is_correct = final_step.reward >= 1.0
        elif hasattr(final_step, 'info') and final_step.info:
            is_correct = final_step.info.get('is_correct', False)
        
        trajectory_data.append({
            'token_count': token_count,
            'is_correct': is_correct,
            'task_id': getattr(result, 'task_id', i),
            'conversation': conversation_text[:200] + "..." if len(conversation_text) > 200 else conversation_text
        })
    
    if not trajectory_data:
        print("No trajectory data found!")
        return
    
    print(f"Total trajectories analyzed: {len(trajectory_data)}")
    print(f"Max context length: {max_context_length}")
    
    # Calculate basic statistics
    token_counts = [t['token_count'] for t in trajectory_data]
    correct_count = sum(1 for t in trajectory_data if t['is_correct'])
    
    print(f"\nBasic Statistics:")
    print(f"  Overall accuracy: {correct_count}/{len(trajectory_data)} ({100*correct_count/len(trajectory_data):.1f}%)")
    print(f"  Token count - Min: {min(token_counts)}, Max: {max(token_counts)}, Mean: {np.mean(token_counts):.1f}, Median: {np.median(token_counts):.1f}")
    
    # Analyze at different context length cutoffs
    # cutoff_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cutoff_percentages = [6.25*i for i in range(16)]
    total_trajectories = len(trajectory_data)
    
    print(f"\nContext Length Cutoff Analysis:")
    print(f"{'Cutoff %':<10} {'Max Tokens':<12} {'Valid Trajs':<12} {'Valid Correct':<13} {'Accuracy':<12} {'Coverage':<12} {'Truncated':<12}")
    print("-" * 100)
    
    results_summary = []
    
    for cutoff_pct in cutoff_percentages:
        cutoff_tokens = int(max_context_length * cutoff_pct / 100)
        
        # Count valid trajectories (those that fit within cutoff)
        valid_trajectories = [t for t in trajectory_data if t['token_count'] <= cutoff_tokens]
        truncated_count = len(trajectory_data) - len(valid_trajectories)
        
        if valid_trajectories:
            correct_in_valid = sum(1 for t in valid_trajectories if t['is_correct'])
            accuracy = 100 * correct_in_valid / len(valid_trajectories)  # Accuracy within valid trajectories
        else:
            correct_in_valid = 0
            accuracy = 0.0
        
        # Coverage: percentage of ALL trajectories that are correct at this cutoff
        coverage = 100 * correct_in_valid / total_trajectories
        
        print(f"{cutoff_pct:<10} {cutoff_tokens:<12} {len(valid_trajectories):<12} {correct_in_valid:<13} {accuracy:<11.1f}% {coverage:<11.1f}% {truncated_count:<12}")
        
        results_summary.append({
            'cutoff_pct': cutoff_pct,
            'cutoff_tokens': cutoff_tokens,
            'valid_count': len(valid_trajectories),
            'correct_count': correct_in_valid,
            'accuracy': accuracy,
            'coverage': coverage,
            'truncated_count': truncated_count
        })
    
    # Find optimal context length and coverage analysis
    print(f"\nContext Length Insights:")
    max_accuracy = max(r['accuracy'] for r in results_summary)
    max_coverage = max(r['coverage'] for r in results_summary)
    final_coverage = results_summary[-1]['coverage']  # Coverage at 100%
    
    optimal_cutoffs = [r for r in results_summary if r['accuracy'] == max_accuracy]
    
    if optimal_cutoffs:
        min_optimal = min(optimal_cutoffs, key=lambda x: x['cutoff_tokens'])
        print(f"  Maximum accuracy achieved: {max_accuracy:.1f}%")
        print(f"  Minimum context length for max accuracy: {min_optimal['cutoff_tokens']} tokens ({min_optimal['cutoff_pct']}%)")
    
    print(f"  Final coverage (at 100% context): {final_coverage:.1f}%")
    print(f"  Maximum coverage achieved: {max_coverage:.1f}%")
    
    # Show coverage progression
    print(f"\nCoverage Progression:")
    for r in results_summary[::2]:  # Show every other cutoff for brevity
        print(f"  {r['cutoff_pct']:>3}% context: {r['coverage']:>5.1f}% coverage ({r['correct_count']}/{total_trajectories} correct)")
    
    # Show token distribution
    print(f"\nToken Count Distribution:")
    # percentiles = [25, 50, 75, 90, 95, 99]
    percentiles = [6.25*i for i in range(16)]
    for p in percentiles:
        value = np.percentile(token_counts, p)
        print(f"  {p}th percentile: {value:.0f} tokens")
    
    # Show some example trajectories at different lengths
    print(f"\nExample Trajectories by Length:")
    trajectory_data_sorted = sorted(trajectory_data, key=lambda x: x['token_count'])
    
    # Show shortest, median, and longest
    indices = [0, len(trajectory_data_sorted)//2, -1]
    labels = ["Shortest", "Median", "Longest"]
    
    for idx, label in zip(indices, labels):
        traj = trajectory_data_sorted[idx]
        status = "✓ Correct" if traj['is_correct'] else "✗ Incorrect"
        print(f"  {label}: {traj['token_count']} tokens - {status}")
        print(f"    Preview: {traj['conversation'][:150]}...")
    
    return results_summary


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    # Use Qwen3-0.6B model as specified
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = countdown_reward_fn

    env_args = {
        "reward_fn": reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=CountDownAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=8192,  # Smaller than deepscaler since countdown tasks are simpler
        max_prompt_length=1024,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("countdown", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_countdown_data import prepare_countdown_data

        _, test_dataset = prepare_countdown_data()

    tasks = test_dataset.repeat(n=1)  # repeat to evaluate pass@k

    print("Running inference on countdown tasks...")
    results = asyncio.run(engine.execute_tasks(tasks))
    
    # Original pass@k computation
    print("\n" + "="*80)
    print("STANDARD EVALUATION")
    print("="*80)
    compute_pass_at_k(results)
    
    # New trajectory statistics analysis
    max_context_length = engine.max_response_length  # Total context budget
    analyze_trajectory_statistics(results, tokenizer, max_context_length)
