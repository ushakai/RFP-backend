-- Migration: Add original_rfp_date to client_match_questions RPC function
-- This enables date-based prioritization in search results
-- Date: 2025-12-18

-- Drop existing function
DROP FUNCTION IF EXISTS client_match_questions(vector, float, int, uuid, uuid);

-- Recreate with original_rfp_date in the return type
CREATE OR REPLACE FUNCTION client_match_questions(
    query_embedding vector(768),
    match_threshold float,
    match_count int,
    p_client_id uuid,
    p_rfp_id uuid DEFAULT NULL
)
RETURNS TABLE (
    question_id uuid,
    question text,
    answer_id uuid,
    answer text,
    similarity float,
    original_rfp_date date
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        q.id as question_id,
        q.original_text as question,
        m.answer_id,
        a.answer_text as answer,
        1 - (q.embedding <=> query_embedding) as similarity,
        r.original_rfp_date
    FROM client_questions q
    LEFT JOIN client_question_answer_mappings m ON q.id = m.question_id
    LEFT JOIN client_answers a ON m.answer_id = a.id
    LEFT JOIN client_rfps r ON q.rfp_id = r.id
    WHERE q.client_id = p_client_id
    AND (p_rfp_id IS NULL OR q.rfp_id = p_rfp_id)
    AND q.embedding IS NOT NULL
    AND 1 - (q.embedding <=> query_embedding) > match_threshold
    ORDER BY q.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Add comment for documentation
COMMENT ON FUNCTION client_match_questions IS 
'Matches questions by vector similarity with optional RFP filtering. Returns original_rfp_date for date-based prioritization.';

