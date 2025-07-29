#!/usr/bin/env python3
"""
Query episodes from the episode store.
This script demonstrates how to retrieve and analyze stored episodes.
"""

import sys
from pathlib import Path

# Add the project root to Python path  
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rllm.db.episode_store import SQLiteEpisodeStore


def main():
    # Connect to the episode store
    db_path = "logs/critique_episodes.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database file '{db_path}' not found.")
        print("   Run 'run_critique_workflow_with_storage.py' first to create episodes.")
        return
    
    episode_store = SQLiteEpisodeStore(db_path)
    
    try:
        # Get overall statistics
        overall_stats = episode_store.get_statistics()
        print("📊 Overall Episode Store Statistics:")
        print(f"   Total episodes: {overall_stats['total_episodes']}")
        print(f"   Correct episodes: {overall_stats['correct_episodes']}")
        print(f"   Overall accuracy: {overall_stats['accuracy']:.2%}")
        
        if overall_stats['total_episodes'] == 0:
            print("\n   No episodes found in the database.")
            return
        
        # Get statistics for a specific workflow
        workflow_id = "critique_math_experiment_001"
        workflow_stats = episode_store.get_statistics(workflow_id)
        print(f"\n📋 Statistics for workflow '{workflow_id}':")
        print(f"   Episodes in this workflow: {workflow_stats['total_episodes']}")
        print(f"   Correct episodes: {workflow_stats['correct_episodes']}")
        print(f"   Workflow accuracy: {workflow_stats['accuracy']:.2%}")
        
        # Retrieve and display some episodes
        episodes = episode_store.get_episodes(workflow_id, limit=10)
        print(f"\n📝 Sample episodes from workflow '{workflow_id}' (showing first {len(episodes)}):")
        
        for i, episode in enumerate(episodes, 1):
            print(f"\n   Episode {i}:")
            print(f"     ID: {episode.id}")
            print(f"     Correct: {'✅' if episode.is_correct else '❌'}")
            print(f"     Termination: {episode.termination_reason.value if episode.termination_reason else 'None'}")
            
            if episode.task:
                # Show truncated question
                question = episode.task.get('question', 'N/A')
                if len(question) > 100:
                    question = question[:100] + "..."
                print(f"     Question: {question}")
        
        # Show workflow summary
        print(f"\n🎯 Workflow Summary:")
        if workflow_stats['total_episodes'] > 0:
            success_rate = workflow_stats['accuracy']
            if success_rate >= 0.8:
                rating = "Excellent! 🌟"
            elif success_rate >= 0.6:
                rating = "Good 👍"
            elif success_rate >= 0.4:
                rating = "Needs improvement 📈"
            else:
                rating = "Poor performance 📉"
            
            print(f"   Performance: {rating}")
            print(f"   Success rate: {success_rate:.1%}")
        
    finally:
        episode_store.close()


if __name__ == "__main__":
    main() 