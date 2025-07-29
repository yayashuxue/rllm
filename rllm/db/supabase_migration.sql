-- Supabase Migration for rLLM Episode Store
-- Run this SQL in your Supabase SQL Editor to create the required tables

-- Enable Row Level Security (RLS) if desired
-- You can customize the policies based on your security requirements

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create episodes table
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    task_data JSONB,
    is_correct BOOLEAN NOT NULL DEFAULT FALSE,
    termination_reason TEXT,
    trajectories_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_episodes_workflow_id ON episodes(workflow_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_is_correct ON episodes(is_correct);
CREATE INDEX IF NOT EXISTS idx_episodes_workflow_created ON episodes(workflow_id, created_at DESC);

-- Create a composite index for statistics queries
CREATE INDEX IF NOT EXISTS idx_episodes_workflow_correct ON episodes(workflow_id, is_correct);

-- Optional: Enable Row Level Security (RLS)
-- Uncomment the following if you want to enable RLS for additional security

-- ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (customize based on your needs):

-- Allow authenticated users to read all workflows
-- CREATE POLICY "Users can view workflows" ON workflows
--     FOR SELECT USING (auth.role() = 'authenticated');

-- Allow authenticated users to insert workflows
-- CREATE POLICY "Users can insert workflows" ON workflows
--     FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Allow authenticated users to read all episodes
-- CREATE POLICY "Users can view episodes" ON episodes
--     FOR SELECT USING (auth.role() = 'authenticated');

-- Allow authenticated users to insert episodes
-- CREATE POLICY "Users can insert episodes" ON episodes
--     FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Allow authenticated users to update episodes
-- CREATE POLICY "Users can update episodes" ON episodes
--     FOR UPDATE USING (auth.role() = 'authenticated');

-- Grant permissions to authenticated users (if using RLS)
-- GRANT SELECT, INSERT, UPDATE ON workflows TO authenticated;
-- GRANT SELECT, INSERT, UPDATE ON episodes TO authenticated;

-- Create a function to get episode statistics (optional helper)
CREATE OR REPLACE FUNCTION get_episode_stats(target_workflow_id TEXT DEFAULT NULL)
RETURNS TABLE(
    total_episodes BIGINT,
    correct_episodes BIGINT,
    accuracy NUMERIC(5,4)
) 
LANGUAGE plpgsql
AS $$
BEGIN
    IF target_workflow_id IS NULL THEN
        -- Get overall statistics
        RETURN QUERY
        SELECT 
            COUNT(*)::BIGINT as total_episodes,
            COUNT(CASE WHEN is_correct THEN 1 END)::BIGINT as correct_episodes,
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC, 4)
                ELSE 0.0
            END as accuracy
        FROM episodes;
    ELSE
        -- Get statistics for specific workflow
        RETURN QUERY
        SELECT 
            COUNT(*)::BIGINT as total_episodes,
            COUNT(CASE WHEN is_correct THEN 1 END)::BIGINT as correct_episodes,
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    ROUND(COUNT(CASE WHEN is_correct THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC, 4)
                ELSE 0.0
            END as accuracy
        FROM episodes 
        WHERE workflow_id = target_workflow_id;
    END IF;
END;
$$;

-- Example usage of the stats function:
-- SELECT * FROM get_episode_stats(); -- Overall stats
-- SELECT * FROM get_episode_stats('my_workflow_id'); -- Workflow-specific stats

COMMENT ON TABLE workflows IS 'Stores workflow metadata for organizing episodes';
COMMENT ON TABLE episodes IS 'Stores individual episode data with trajectories and results';
COMMENT ON FUNCTION get_episode_stats(TEXT) IS 'Helper function to calculate episode statistics';

-- Confirm tables were created
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE tablename IN ('workflows', 'episodes')
ORDER BY tablename; 