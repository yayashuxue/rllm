# Episode Store

The Episode Store provides a way to persist episode objects to a database whenever tasks finish execution in the `AgentWorkflowEngine`. This enables tracking, analysis, and comparison of different workflow runs.

## Features

- **Workflow Grouping**: Episodes are organized by `workflow_id`, allowing you to group related experiments
- **SQLite Backend**: Episodes are stored in a local SQLite database for easy access and analysis
- **Automatic Storage**: Episodes are automatically stored when `workflow_id` is provided
- **Statistics**: Built-in methods to analyze success rates and episode counts
- **Thread-Safe**: Multiple workflows can safely write to the same database

## Quick Start

### Basic Usage

```python
from rllm import AgentWorkflowEngine, SQLiteEpisodeStore

# Option 1: Auto-initialize episode store (recommended - simplest)
engine = AgentWorkflowEngine(
    workflow_cls=MyWorkflow,
    workflow_args=workflow_args,
    rollout_engine=rollout_engine,
    default_db_path="experiments/my_episodes.db",  # Auto-creates SQLiteEpisodeStore
    # ... other arguments
)

# Option 2: Manual episode store creation (if you need custom configuration)
# episode_store = SQLiteEpisodeStore("my_episodes.db")
# engine = AgentWorkflowEngine(..., episode_store=episode_store)

# Execute tasks with workflow_id - episodes automatically stored
results = await engine.execute_tasks(
    tasks=tasks,
    workflow_id="my_experiment_001"  # Episodes grouped by this ID
)

# Access the episode store through the engine
stats = engine.episode_store.get_statistics("my_experiment_001")

# Clean up
engine.shutdown()  # Also closes episode store
```

### Querying Stored Episodes

```python
from rllm import SQLiteEpisodeStore

# Connect to existing database
store = SQLiteEpisodeStore("my_episodes.db")

# Get overall statistics
stats = store.get_statistics()
print(f"Total episodes: {stats['total_episodes']}")
print(f"Overall accuracy: {stats['accuracy']:.2%}")

# Get statistics for specific workflow
workflow_stats = store.get_statistics("my_experiment_001")
print(f"Workflow accuracy: {workflow_stats['accuracy']:.2%}")

# Retrieve episodes
episodes = store.get_episodes("my_experiment_001", limit=10)
for episode in episodes:
    print(f"Episode {episode.id}: {'✅' if episode.is_correct else '❌'}")

store.close()
```

## Classes

### EpisodeStore (Abstract Base Class)

The base interface for episode storage implementations.

```python
class EpisodeStore(ABC):
    @abstractmethod
    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """Store an episode with associated workflow_id."""
        
    @abstractmethod
    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> List[Episode]:
        """Retrieve episodes for a given workflow_id."""
        
    @abstractmethod
    def close(self) -> None:
        """Close the store connection."""
```

### SQLiteEpisodeStore

SQLite-backed implementation of EpisodeStore.

```python
store = SQLiteEpisodeStore("path/to/database.db")
```

**Features:**
- Automatic database schema creation
- JSON serialization of episode data
- Indexed queries by workflow_id
- Thread-safe operations

### NoOpEpisodeStore

A no-operation store that doesn't actually persist anything. This is the default when no episode store is provided.

```python
store = NoOpEpisodeStore()  # Does nothing
```

## Database Schema

The SQLite implementation uses two tables:

### workflows table
- `id` (TEXT PRIMARY KEY): The workflow identifier
- `created_at` (TIMESTAMP): When the workflow was first seen

### episodes table  
- `id` (TEXT PRIMARY KEY): The episode identifier
- `workflow_id` (TEXT): Foreign key to workflows table
- `task_data` (TEXT): JSON-serialized task information
- `is_correct` (BOOLEAN): Whether the episode was marked as correct
- `termination_reason` (TEXT): How the episode ended
- `trajectories_data` (TEXT): JSON-serialized trajectory information
- `created_at` (TIMESTAMP): When the episode was stored

## Workflow Integration

### AgentWorkflowEngine Changes

The `AgentWorkflowEngine` now automatically creates an episode store if none is provided:

```python
# Auto-initialization (recommended)
engine = AgentWorkflowEngine(
    # ... existing parameters
    default_db_path="path/to/episodes.db",  # Optional: defaults to "episodes.db"
)

# Manual store creation (for custom configuration)
engine = AgentWorkflowEngine(
    # ... existing parameters
    episode_store=custom_episode_store,  # Optional: overrides auto-initialization
)
```

### Method Updates

Both execution methods now accept an optional `workflow_id`:

```python
# For regular execution
results = await engine.execute_tasks(tasks, workflow_id="experiment_001")

# For VERL integration  
results = await engine.execute_tasks_verl(batch, workflow_id="training_run_001")
```

When `workflow_id` is provided, all episodes from that execution will be automatically stored in the episode store.

## Best Practices

### Workflow ID Naming

Use descriptive, unique workflow IDs:

```python
# Good examples
workflow_id = "math_critique_baseline_v1"
workflow_id = "code_generation_experiment_2024_01_15"
workflow_id = f"hyperparameter_sweep_temp_{temperature}_top_p_{top_p}"

# Avoid generic names
workflow_id = "test"  # Too generic
workflow_id = "run1"  # Not descriptive
```

### Resource Management

Always properly close resources:

```python
try:
    engine = AgentWorkflowEngine(episode_store=store, ...)
    results = await engine.execute_tasks(tasks, workflow_id="exp1")
finally:
    engine.shutdown()  # Automatically closes episode store
    
# Or explicitly:
store.close()
```

### Database Location

Consider where to store your database:

```python
# Local development
store = SQLiteEpisodeStore("experiments.db")

# Organized by date
store = SQLiteEpisodeStore(f"experiments_{datetime.now().strftime('%Y_%m_%d')}.db")

# Project-specific location
store = SQLiteEpisodeStore("logs/episode_store.db")
```

## Examples

See the example scripts for complete usage:

- `examples/workflow/critique/run_critique_workflow_with_storage.py` - Shows how to use episode storage
- `examples/workflow/critique/query_episode_store.py` - Demonstrates querying stored episodes

## Limitations

- **Trajectory Reconstruction**: The current implementation stores trajectory data as JSON but doesn't fully reconstruct `Trajectory` objects when retrieving (this could be enhanced)
- **SQLite Only**: Currently only SQLite backend is implemented (could be extended to other databases)
- **No Migration**: Database schema is fixed (versioning could be added in the future)

## Future Enhancements

Potential improvements could include:

- Full trajectory deserialization
- Multiple database backends (PostgreSQL, MongoDB, etc.)
- Episode labeling and tagging
- Advanced querying and filtering
- Database schema versioning
- Episode comparison and diffing tools 