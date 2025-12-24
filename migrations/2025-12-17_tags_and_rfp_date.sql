-- Migration: Add normalized tags schema and original_rfp_date
-- Created: 2025-12-17
-- Description: Introduces tags table, rfp_tags join table, and original_rfp_date column for scalable tag management

-- 1. Create tags table
-- This table stores unique tag names per client for reusability
CREATE TABLE IF NOT EXISTS tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(client_id, name) -- Ensure tag names are unique per client
);

CREATE INDEX IF NOT EXISTS idx_tags_client_id ON tags(client_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

-- 2. Create rfp_tags join table
-- This table links RFPs to tags (many-to-many relationship)
CREATE TABLE IF NOT EXISTS rfp_tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rfp_id UUID NOT NULL REFERENCES client_rfps(id) ON DELETE CASCADE,
    tag_id UUID NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(rfp_id, tag_id) -- Prevent duplicate tag assignments
);

CREATE INDEX IF NOT EXISTS idx_rfp_tags_rfp_id ON rfp_tags(rfp_id);
CREATE INDEX IF NOT EXISTS idx_rfp_tags_tag_id ON rfp_tags(tag_id);

-- 3. Add original_rfp_date column to client_rfps
-- This stores the original date of the RFP as provided by the user during import
ALTER TABLE client_rfps 
ADD COLUMN IF NOT EXISTS original_rfp_date DATE;

CREATE INDEX IF NOT EXISTS idx_client_rfps_original_date ON client_rfps(original_rfp_date);

-- 4. Add trigger to maintain updated_at on tags table
CREATE OR REPLACE FUNCTION update_tags_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_tags_updated_at ON tags;
CREATE TRIGGER trigger_update_tags_updated_at
    BEFORE UPDATE ON tags
    FOR EACH ROW
    EXECUTE FUNCTION update_tags_updated_at();

-- 5. Add comments for documentation
COMMENT ON TABLE tags IS 'Stores reusable tag names per client for RFP categorization';
COMMENT ON TABLE rfp_tags IS 'Join table linking RFPs to tags (many-to-many)';
COMMENT ON COLUMN client_rfps.original_rfp_date IS 'Original date of the RFP as specified by user during import';

