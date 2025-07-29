import sqlite3
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from datetime import datetime

from rllm.agents.agent import Episode

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class EpisodeStore(ABC):
    """Abstract interface for storing episodes."""
    
    @abstractmethod
    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """Store an episode with associated workflow_id."""
        pass
    
    @abstractmethod
    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> List[Episode]:
        """Retrieve episodes for a given workflow_id."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the store connection."""
        pass


class SQLiteEpisodeStore(EpisodeStore):
    """SQLite-backed episode store implementation."""
    
    def __init__(self, db_path: str):
        """Initialize SQLite episode store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_data TEXT,
                is_correct BOOLEAN,
                termination_reason TEXT,
                trajectories_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_workflow_id 
            ON episodes(workflow_id)
        """)
        
        self.conn.commit()
    
    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """Store an episode with associated workflow_id."""
        cursor = self.conn.cursor()
        
        # Insert workflow if it doesn't exist
        cursor.execute("""
            INSERT OR IGNORE INTO workflows (id) VALUES (?)
        """, (workflow_id,))
        
        # Serialize trajectories to JSON
        trajectories_json = {}
        for name, trajectory in episode.trajectories.items():
            trajectories_json[name] = trajectory.to_dict()
        
        # Insert episode
        cursor.execute("""
            INSERT OR REPLACE INTO episodes 
            (id, workflow_id, task_data, is_correct, termination_reason, trajectories_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            episode.id,
            workflow_id,
            json.dumps(episode.task) if episode.task else None,
            episode.is_correct,
            episode.termination_reason.value if episode.termination_reason else None,
            json.dumps(trajectories_json)
        ))
        
        self.conn.commit()
    
    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> List[Episode]:
        """Retrieve episodes for a given workflow_id."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT id, task_data, is_correct, termination_reason, trajectories_data
            FROM episodes 
            WHERE workflow_id = ?
            ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (workflow_id,))
        rows = cursor.fetchall()
        
        episodes = []
        for row in rows:
            episode_id, task_data, is_correct, termination_reason, trajectories_data = row
            
            # Deserialize data
            task = json.loads(task_data) if task_data else None
            trajectories_dict = json.loads(trajectories_data) if trajectories_data else {}
            
            # Reconstruct episode (note: this is a simplified reconstruction)
            # For full reconstruction, you'd need to properly deserialize trajectories
            episode = Episode(
                id=episode_id,
                task=task,
                is_correct=bool(is_correct),
                trajectories={}  # Simplified - would need proper trajectory reconstruction
            )
            
            # Set termination reason if available
            if termination_reason:
                from rllm.workflows.workflow import TerminationReason
                episode.termination_reason = TerminationReason(termination_reason)
            
            episodes.append(episode)
        
        return episodes
    
    def get_statistics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for episodes in the store."""
        cursor = self.conn.cursor()
        
        if workflow_id:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_episodes
                FROM episodes 
                WHERE workflow_id = ?
            """, (workflow_id,))
        else:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_episodes
                FROM episodes
            """)
        
        total, correct = cursor.fetchone()
        accuracy = (correct / total) if total > 0 else 0.0
        
        return {
            "total_episodes": total,
            "correct_episodes": correct,
            "accuracy": accuracy
        }
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


class NoOpEpisodeStore(EpisodeStore):
    """No-operation episode store that doesn't actually store anything."""
    
    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """No-op implementation."""
        pass
    
    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> List[Episode]:
        """No-op implementation."""
        return []
    
    def close(self) -> None:
        """No-op implementation."""
        pass


class SupabaseEpisodeStore(EpisodeStore):
    """Supabase-backed episode store implementation."""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """Initialize Supabase episode store.
        
        Args:
            supabase_url: Supabase project URL. If None, reads from SUPABASE_URL env var.
            supabase_key: Supabase anon key. If None, reads from SUPABASE_KEY env var.
        
        Raises:
            ImportError: If supabase package is not installed
            ValueError: If credentials are not provided
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "Supabase package not installed. Install with: pip install supabase"
            )
        
        # Get credentials from parameters or environment variables
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase credentials required. Provide supabase_url and supabase_key "
                "or set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        # Create Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize database schema if needed
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema.
        
        Note: This assumes the tables already exist in Supabase.
        Run the SQL migration script to create them if needed.
        """
        # Check if tables exist by trying to query them
        try:
            # Test workflows table
            result = self.client.table("workflows").select("id").limit(1).execute()
            
            # Test episodes table  
            result = self.client.table("episodes").select("id").limit(1).execute()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not verify Supabase tables exist: {e}")
            print("📋 Please ensure you've run the Supabase migration to create:")
            print("   - workflows table")
            print("   - episodes table")
            print("📄 See rllm/db/supabase_migration.sql for the schema")
    
    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """Store an episode with associated workflow_id."""
        try:
            # Validate inputs
            if not workflow_id or not workflow_id.strip():
                raise ValueError("workflow_id cannot be empty")
            if not episode.id or not episode.id.strip():
                raise ValueError("episode.id cannot be empty")
            
            # Use proper ISO format with timezone for Supabase TIMESTAMPTZ
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Insert workflow if it doesn't exist (upsert)
            workflow_data = {
                "id": workflow_id.strip(),
                "created_at": timestamp
            }
            
            try:
                self.client.table("workflows").upsert(workflow_data, on_conflict="id").execute()
            except Exception as e:
                print(f"❌ Failed to upsert workflow {workflow_id}: {e}")
                print(f"   Workflow data: {workflow_data}")
                raise
            
            # Serialize trajectories to JSON with error handling
            trajectories_json = {}
            try:
                for name, trajectory in episode.trajectories.items():
                    traj_dict = trajectory.to_dict()
                    # Ensure all values are JSON serializable
                    trajectories_json[name] = traj_dict
                
                # Test JSON serialization
                json.dumps(trajectories_json)
                
            except Exception as e:
                print(f"❌ Failed to serialize trajectories: {e}")
                print(f"   Trajectory keys: {list(episode.trajectories.keys())}")
                # Use empty dict as fallback
                trajectories_json = {}
            
            # Prepare episode data with validation
            episode_data = {
                "id": episode.id.strip(),
                "workflow_id": workflow_id.strip(),
                "task_data": json.dumps(episode.task) if episode.task else None,
                "is_correct": bool(episode.is_correct),
                "termination_reason": episode.termination_reason.value if episode.termination_reason else None,
                "trajectories_data": json.dumps(trajectories_json),
                "created_at": timestamp
            }
            
            # Validate episode data can be JSON serialized
            try:
                json.dumps(episode_data)
            except Exception as e:
                print(f"❌ Episode data is not JSON serializable: {e}")
                print(f"   Episode data keys: {list(episode_data.keys())}")
                raise
            
            # Insert episode (upsert to handle duplicates)
            try:
                self.client.table("episodes").upsert(episode_data, on_conflict="id").execute()
            except Exception as e:
                print(f"❌ Failed to upsert episode {episode.id}: {e}")
                print(f"   Episode data: {episode_data}")
                raise
            
        except Exception as e:
            print(f"❌ Error storing episode {getattr(episode, 'id', 'unknown')} to Supabase: {e}")
            raise
    
    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> List[Episode]:
        """Retrieve episodes for a given workflow_id."""
        try:
            # Build query
            query = (
                self.client.table("episodes")
                .select("id, task_data, is_correct, termination_reason, trajectories_data")
                .eq("workflow_id", workflow_id)
                .order("created_at", desc=True)
            )
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            
            episodes = []
            for row in result.data:
                # Deserialize data
                task = json.loads(row["task_data"]) if row["task_data"] else None
                
                # Reconstruct episode (simplified reconstruction)
                episode = Episode(
                    id=row["id"],
                    task=task,
                    is_correct=bool(row["is_correct"]),
                    trajectories={}  # Simplified - would need proper trajectory reconstruction
                )
                
                # Set termination reason if available
                if row["termination_reason"]:
                    from rllm.workflows.workflow import TerminationReason
                    episode.termination_reason = TerminationReason(row["termination_reason"])
                
                episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            print(f"❌ Error retrieving episodes for workflow {workflow_id} from Supabase: {e}")
            return []
    
    def get_statistics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for episodes in the store."""
        try:
            if workflow_id:
                # Get stats for specific workflow
                result = (
                    self.client.table("episodes")
                    .select("is_correct")
                    .eq("workflow_id", workflow_id)
                    .execute()
                )
            else:
                # Get overall stats
                result = (
                    self.client.table("episodes")
                    .select("is_correct")
                    .execute()
                )
            
            episodes = result.data
            total = len(episodes)
            correct = sum(1 for ep in episodes if ep["is_correct"])
            accuracy = (correct / total) if total > 0 else 0.0
            
            return {
                "total_episodes": total,
                "correct_episodes": correct,
                "accuracy": accuracy
            }
            
        except Exception as e:
            print(f"❌ Error getting statistics from Supabase: {e}")
            return {
                "total_episodes": 0,
                "correct_episodes": 0,
                "accuracy": 0.0
            }
    
    def close(self) -> None:
        """Close the Supabase connection."""
        # Supabase client doesn't need explicit closing
        # Just clear the reference
        self.client = None 