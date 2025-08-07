#!/usr/bin/env python3
"""
Example client for the rLLM Workflow API
Demonstrates both synchronous and asynchronous task execution
"""

import time
import requests
from typing import Dict, List, Any


class RLLMClient:
    """Simple client for the rLLM Workflow API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check if the API server is running"""
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def list_workflows(self) -> dict:
        """List available workflow types"""
        response = requests.get(f"{self.base_url}/workflows")
        response.raise_for_status()
        return response.json()
    
    def execute_task_sync(self, task: Dict[str, Any], workflow_type: str = "single_turn") -> dict:
        """Execute a single task synchronously"""
        response = requests.post(
            f"{self.base_url}/execute",
            json={
                "task": task,
                "workflow_type": workflow_type
            }
        )
        response.raise_for_status()
        return response.json()
    
    def execute_batch_sync(self, tasks: List[Dict[str, Any]], workflow_type: str = "single_turn") -> dict:
        """Execute multiple tasks in batch synchronously"""
        response = requests.post(
            f"{self.base_url}/execute_batch",
            json={
                "tasks": tasks,
                "workflow_type": workflow_type
            }
        )
        response.raise_for_status()
        return response.json()
    
    def submit_task_async(self, task: Dict[str, Any], workflow_type: str = "single_turn") -> str:
        """Submit task for asynchronous execution"""
        response = requests.post(
            f"{self.base_url}/submit",
            json={
                "task": task,
                "workflow_type": workflow_type
            }
        )
        response.raise_for_status()
        return response.json()["task_id"]
    
    def get_task_status(self, task_id: str) -> dict:
        """Get task status"""
        response = requests.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_task_results(self, task_id: str) -> dict:
        """Get task results"""
        response = requests.get(f"{self.base_url}/results/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: float = 1.0) -> dict:
        """Wait for a task to complete and return the results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            print(f"Task {task_id}: {status['status']}")
            
            if status["status"] == "completed":
                return self.get_task_results(task_id)
            elif status["status"] == "failed":
                raise Exception(f"Task failed: {status.get('result', {}).get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def list_all_tasks(self) -> dict:
        """List all tasks and their statuses"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()
    
    def delete_task(self, task_id: str) -> dict:
        """Delete a task and its results"""
        response = requests.delete(f"{self.base_url}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()


def create_math_task(question: str, answer: str, idx: int = 0) -> Dict[str, Any]:
    """Helper function to create a math task"""
    return {
        "question": question,
        "ground_truth": answer,
        "idx": idx,
        "data_source": "math"
    }


def demo_synchronous_execution(client: RLLMClient):
    """Demonstrate synchronous task execution"""
    print("\n=== Synchronous Execution Demo ===")
    
    # Create a simple math task
    task = create_math_task("What is 15 + 27?", "42")
    
    print(f"Executing task: {task['question']}")
    
    # Execute with single_turn workflow
    result = client.execute_task_sync(task, "single_turn")
    
    print(f"Task ID: {result['task_id']}")
    print(f"Status: {result['status']}")
    print(f"Correct: {result['episode']['is_correct']}")
    
    # Show solver's final answer if available
    if "solver" in result["episode"]["trajectories"]:
        solver_steps = result["episode"]["trajectories"]["solver"]["steps"]
        if solver_steps:
            final_action = solver_steps[-1].get("action", {})
            if hasattr(final_action, 'action'):
                print(f"Solver's answer: {final_action.action}")


def demo_batch_execution(client: RLLMClient):
    """Demonstrate batch task execution"""
    print("\n=== Batch Execution Demo ===")
    
    # Create multiple math tasks
    tasks = [
        create_math_task("What is 8 + 7?", "15", 0),
        create_math_task("What is 12 * 3?", "36", 1),
        create_math_task("What is 100 / 4?", "25", 2),
    ]
    
    print(f"Executing {len(tasks)} tasks in batch...")
    
    # Execute batch with single_turn workflow
    results = client.execute_batch_sync(tasks, "single_turn")
    
    print(f"Batch Status: {results['status']}")
    print("Results:")
    for i, episode in enumerate(results["episodes"]):
        task_question = tasks[i]["question"]
        is_correct = episode["is_correct"]
        print(f"  Task {i}: {task_question} -> Correct: {is_correct}")


def demo_asynchronous_execution(client: RLLMClient):
    """Demonstrate asynchronous task execution"""
    print("\n=== Asynchronous Execution Demo ===")
    
    # Create a complex task
    task = create_math_task(
        "A rectangle has a length of 12 cm and a width of 8 cm. What is its area and perimeter?",
        "Area: 96 cm², Perimeter: 40 cm"
    )
    
    print(f"Submitting task asynchronously: {task['question']}")
    
    # Submit task for async execution
    task_id = client.submit_task_async(task, "multi_turn")
    print(f"Task submitted with ID: {task_id}")
    
    # Wait for completion
    try:
        result = client.wait_for_task(task_id, timeout=120)
        print(f"Task completed! Correct: {result['is_correct']}")
        
        # Clean up
        client.delete_task(task_id)
        print(f"Task {task_id} deleted")
        
    except (TimeoutError, Exception) as e:
        print(f"Task execution failed: {e}")


def demo_workflow_comparison(client: RLLMClient):
    """Compare different workflow types on the same task"""
    print("\n=== Workflow Comparison Demo ===")
    
    task = create_math_task("What is the square root of 144?", "12")
    
    workflows = ["single_turn", "multi_turn"]
    
    print(f"Testing task with different workflows: {task['question']}")
    
    for workflow_type in workflows:
        try:
            print(f"\nTesting with {workflow_type} workflow...")
            result = client.execute_task_sync(task, workflow_type)
            is_correct = result["episode"]["is_correct"]
            print(f"  Result: Correct = {is_correct}")
            
        except Exception as e:
            print(f"  Error with {workflow_type}: {e}")


def main():
    """Main demonstration function"""
    # Initialize client
    client = RLLMClient()
    
    try:
        # Health check
        print("Checking API health...")
        health = client.health_check()
        print(f"API Status: {health['message']}")
        
        # List available workflows
        workflows = client.list_workflows()
        print(f"Available workflows: {workflows['workflows']}")
        
        # Run demonstrations
        demo_synchronous_execution(client)
        demo_batch_execution(client)
        demo_asynchronous_execution(client)
        demo_workflow_comparison(client)
        
        # Show all tasks
        print("\n=== Current Tasks ===")
        all_tasks = client.list_all_tasks()
        if all_tasks["tasks"]:
            for task_info in all_tasks["tasks"]:
                print(f"Task {task_info['task_id']}: {task_info['status']}")
        else:
            print("No active tasks")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running with: python api_server.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 