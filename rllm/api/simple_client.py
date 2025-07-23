#!/usr/bin/env python3
"""
Simple client script for quick testing of the rLLM Workflow API
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_single_task():
    """Test executing a single task"""
    task = {
        "question": "What is 2 + 2? Let's think step by step, put your final answer within \\boxed{}.",
        "ground_truth": "4",
        "idx": 0,
        "data_source": "math"
    }
    
    print("Testing single task execution...")
    print(f"Question: {task['question']}")
    
    response = requests.post(
        f"{BASE_URL}/execute",
        json={
            "task": task,
            "workflow_type": "critique"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Task completed successfully")
        print(f"📊 Correct: {result['episode']['is_correct']}")
        print(f"🆔 Task ID: {result['task_id']}")
        
        # Show trajectory info
        episode = result['episode']
        print(f"🔄 Termination reason: {episode.get('termination_reason', 'N/A')}")
        
        # Show agent trajectories
        for agent_name, trajectory in episode['trajectories'].items():
            print(f"🤖 Agent '{agent_name}': {len(trajectory['steps'])} steps, reward: {trajectory['reward']}")
        
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python simple_client.py")
        print("This script tests the rLLM Workflow API with a simple math problem.")
        print("Make sure the API server is running on http://localhost:8000")
        return
    
    try:
        # Health check
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server health check failed")
            return
            
        # Test single task
        test_single_task()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running with: python api_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 