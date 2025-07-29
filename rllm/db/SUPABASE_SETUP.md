# Supabase Setup Guide for rLLM Episode Storage

This guide walks you through setting up Supabase as a cloud-based episode store for rLLM.

## Why Supabase?

- **☁️ Cloud-Based**: Access your episodes from anywhere
- **🚀 Real-Time**: Built-in real-time subscriptions and updates
- **🔐 Secure**: Row-level security and authentication built-in
- **📊 Dashboard**: Beautiful web interface to view and query data
- **🔧 PostgreSQL**: Full power of PostgreSQL with JSON support
- **📈 Scalable**: Auto-scaling infrastructure

## Prerequisites

1. **Install Supabase Python client**:
   ```bash
   pip install supabase
   ```

2. **Create a Supabase project** at [https://supabase.com](https://supabase.com)

## Step-by-Step Setup

### 1. Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up/login and click "New Project"
3. Choose your organization and provide:
   - **Project Name**: e.g., "rllm-episodes"
   - **Database Password**: Choose a strong password
   - **Region**: Choose closest to your location
4. Click "Create new project" and wait for setup to complete

### 2. Get Your Credentials

1. In your Supabase dashboard, go to **Settings > API**
2. Copy these values:
   - **Project URL**: `https://your-project-id.supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### 3. Run Database Migration

1. In your Supabase dashboard, go to **SQL Editor**
2. Create a new query
3. Copy and paste the contents of `rllm/db/supabase_migration.sql`
4. Click "Run" to create the tables and indexes

The migration creates:
- `workflows` table for workflow metadata
- `episodes` table for episode data with JSON trajectories
- Indexes for optimal query performance
- Helper functions for statistics

### 4. Set Environment Variables

Add these to your shell profile (`.bashrc`, `.zshrc`, etc.) or `.env` file:

```bash
export SUPABASE_URL="https://your-project-id.supabase.co"
export SUPABASE_KEY="your-anon-public-key"
```

Or create a `.env` file in your project:
```bash
# .env file
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key
```

### 5. Test the Connection

```python
from rllm import SupabaseEpisodeStore

# Test connection
try:
    store = SupabaseEpisodeStore()
    print("✅ Connected to Supabase successfully!")
    stats = store.get_statistics()
    print(f"📊 Current episodes in database: {stats['total_episodes']}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## Usage Examples

### Basic Usage with AgentWorkflowEngine

```python
from rllm import AgentWorkflowEngine, SupabaseEpisodeStore

# Create Supabase episode store
episode_store = SupabaseEpisodeStore()

# Use with AgentWorkflowEngine
engine = AgentWorkflowEngine(
    workflow_cls=MyWorkflow,
    rollout_engine=rollout_engine,
    episode_store=episode_store,  # Episodes stored in Supabase
    # ... other parameters
)

# Execute tasks - episodes automatically stored in cloud
results = await engine.execute_tasks(
    tasks=tasks,
    workflow_id="my_experiment_001"
)
```

### Direct Usage

```python
from rllm import SupabaseEpisodeStore

# Initialize store
store = SupabaseEpisodeStore()

# Get statistics for a workflow
stats = store.get_statistics("my_experiment_001")
print(f"Accuracy: {stats['accuracy']:.2%}")

# Retrieve recent episodes
episodes = store.get_episodes("my_experiment_001", limit=10)
for episode in episodes:
    print(f"Episode {episode.id}: {'✅' if episode.is_correct else '❌'}")

# Clean up
store.close()
```

### Alternative Credential Methods

```python
# Method 1: Environment variables (recommended)
store = SupabaseEpisodeStore()

# Method 2: Direct parameters
store = SupabaseEpisodeStore(
    supabase_url="https://your-project.supabase.co",
    supabase_key="your-anon-key"
)

# Method 3: Mixed (fallback to env vars)
store = SupabaseEpisodeStore(supabase_url="https://custom-url.supabase.co")
```

## Dashboard Usage

### Viewing Data

1. Go to **Table Editor** in your Supabase dashboard
2. Browse the `workflows` and `episodes` tables
3. Use filters and search to find specific episodes

### Running Queries

In the **SQL Editor**, you can run custom queries:

```sql
-- Get workflow summary
SELECT 
    workflow_id,
    COUNT(*) as total_episodes,
    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy
FROM episodes 
GROUP BY workflow_id
ORDER BY total_episodes DESC;

-- Find recent failures
SELECT id, workflow_id, task_data, termination_reason
FROM episodes 
WHERE is_correct = false 
ORDER BY created_at DESC 
LIMIT 10;

-- Use the helper function
SELECT * FROM get_episode_stats('my_workflow_id');
```

### Real-Time Monitoring

Set up real-time subscriptions to monitor episodes as they're created:

```python
# Example real-time subscription (advanced usage)
def handle_episode_insert(payload):
    episode = payload['new']
    print(f"New episode: {episode['id']} - {'✅' if episode['is_correct'] else '❌'}")

# Subscribe to episode inserts
store.client.table("episodes").on("INSERT", handle_episode_insert).subscribe()
```

## Security Considerations

### Row Level Security (RLS)

For production use, consider enabling RLS:

1. Uncomment the RLS policies in `supabase_migration.sql`
2. Customize policies based on your access requirements
3. Use Supabase Auth for user authentication

### API Key Security

- **anon key**: Safe for client-side use, respects RLS policies
- **service_role key**: Full database access, keep server-side only
- For production: Use anon key with proper RLS policies

## Troubleshooting

### Common Issues

**Connection Errors**:
- Verify your `SUPABASE_URL` and `SUPABASE_KEY`
- Check your internet connection
- Ensure Supabase project is active

**Table Not Found**:
- Run the migration script in SQL Editor
- Check table names are lowercase: `workflows`, `episodes`

**Permission Denied**:
- If using RLS, ensure proper policies are set
- Use service_role key for admin operations (server-side only)

**Import Error**:
```bash
pip install supabase
```

### Getting Help

- **Supabase Docs**: [https://supabase.com/docs](https://supabase.com/docs)
- **Community**: [https://github.com/supabase/supabase/discussions](https://github.com/supabase/supabase/discussions)
- **Status**: [https://status.supabase.com](https://status.supabase.com)

## Migration from SQLite

To migrate existing SQLite episodes to Supabase:

```python
from rllm import SQLiteEpisodeStore, SupabaseEpisodeStore

# Open both stores
sqlite_store = SQLiteEpisodeStore("episodes.db")
supabase_store = SupabaseEpisodeStore()

# Get all episodes from SQLite (this would need custom implementation)
# and transfer to Supabase
# ... migration logic ...
```

## Best Practices

1. **Environment Variables**: Use env vars for credentials
2. **Workflow IDs**: Use descriptive, unique workflow IDs
3. **Indexing**: The migration includes optimized indexes
4. **Monitoring**: Use Supabase dashboard for monitoring
5. **Backups**: Supabase handles backups automatically
6. **Costs**: Monitor usage in Supabase dashboard 