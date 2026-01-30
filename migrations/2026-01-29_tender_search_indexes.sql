-- Migration: Add indexed search columns and lookup tables for fast tender matching
-- Date: 2026-01-29
-- Description: Creates indexed columns for keywords, locations, industries and a lookup table for dropdown suggestions

-- ============================================================================
-- TENDER SEARCH LOOKUP TABLE
-- Stores extracted searchable terms from tenders for dropdown suggestions
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.tender_search_terms (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  term_type text NOT NULL CHECK (term_type IN ('keyword', 'location', 'industry')),
  term_value text NOT NULL,
  term_display text, -- Human-readable display value (e.g., "72" -> "IT Services")
  tender_count integer DEFAULT 0,
  last_seen_at timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now(),
  UNIQUE(term_type, term_value)
);

-- Index for fast lookups by type and value
CREATE INDEX IF NOT EXISTS idx_tender_search_terms_type ON public.tender_search_terms(term_type);
CREATE INDEX IF NOT EXISTS idx_tender_search_terms_value ON public.tender_search_terms(term_value);
CREATE INDEX IF NOT EXISTS idx_tender_search_terms_count ON public.tender_search_terms(term_type, tender_count DESC);

-- ============================================================================
-- ADD INDEXED SEARCH COLUMNS TO TENDERS TABLE
-- These columns will store normalized/extracted values for direct matching
-- ============================================================================

-- Add array columns for indexed search terms (if they don't exist)
DO $$
BEGIN
  -- indexed_keywords: array of extracted keywords from title/description
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_schema = 'public' AND table_name = 'tenders' AND column_name = 'indexed_keywords') THEN
    ALTER TABLE public.tenders ADD COLUMN indexed_keywords text[];
  END IF;

  -- indexed_locations: array of normalized location codes/names
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_schema = 'public' AND table_name = 'tenders' AND column_name = 'indexed_locations') THEN
    ALTER TABLE public.tenders ADD COLUMN indexed_locations text[];
  END IF;

  -- indexed_industries: array of CPV code prefixes (2-digit)
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_schema = 'public' AND table_name = 'tenders' AND column_name = 'indexed_industries') THEN
    ALTER TABLE public.tenders ADD COLUMN indexed_industries text[];
  END IF;

  -- search_text: full-text searchable content (title + description + location)
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_schema = 'public' AND table_name = 'tenders' AND column_name = 'search_text') THEN
    ALTER TABLE public.tenders ADD COLUMN search_text text;
  END IF;
END $$;

-- Create GIN indexes for array containment queries (MUCH faster than scanning)
CREATE INDEX IF NOT EXISTS idx_tenders_indexed_keywords ON public.tenders USING GIN (indexed_keywords);
CREATE INDEX IF NOT EXISTS idx_tenders_indexed_locations ON public.tenders USING GIN (indexed_locations);
CREATE INDEX IF NOT EXISTS idx_tenders_indexed_industries ON public.tenders USING GIN (indexed_industries);

-- Create full-text search index for search_text
CREATE INDEX IF NOT EXISTS idx_tenders_search_text ON public.tenders USING GIN (to_tsvector('english', COALESCE(search_text, '')));

-- ============================================================================
-- RPC FUNCTION FOR FAST INDEXED MATCHING
-- ============================================================================

CREATE OR REPLACE FUNCTION public.match_tenders_indexed(
  p_keywords text[] DEFAULT NULL,
  p_locations text[] DEFAULT NULL,
  p_industries text[] DEFAULT NULL,
  p_limit integer DEFAULT 100,
  p_offset integer DEFAULT 0,
  p_days_lookback integer DEFAULT 30
)
RETURNS TABLE (
  id uuid,
  source text,
  external_id text,
  title text,
  description text,
  summary text,
  deadline timestamptz,
  published_date timestamptz,
  value_amount numeric,
  value_currency text,
  location text,
  category text,
  sector text,
  indexed_keywords text[],
  indexed_locations text[],
  indexed_industries text[],
  match_score float
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  v_cutoff timestamptz;
  v_now timestamptz;
BEGIN
  v_now := now();
  v_cutoff := v_now - (p_days_lookback || ' days')::interval;

  RETURN QUERY
  SELECT 
    t.id,
    t.source,
    t.external_id,
    t.title,
    t.description,
    t.summary,
    t.deadline,
    t.published_date,
    t.value_amount,
    t.value_currency,
    t.location,
    t.category,
    t.sector,
    t.indexed_keywords,
    t.indexed_locations,
    t.indexed_industries,
    -- Calculate match score based on matches
    (
      CASE WHEN p_keywords IS NOT NULL AND array_length(p_keywords, 1) > 0 
           THEN (SELECT COUNT(*) FROM unnest(p_keywords) k WHERE t.indexed_keywords && ARRAY[k])::float / array_length(p_keywords, 1)
           ELSE 0.0 
      END +
      CASE WHEN p_locations IS NOT NULL AND array_length(p_locations, 1) > 0 
           THEN (SELECT COUNT(*) FROM unnest(p_locations) l WHERE t.indexed_locations && ARRAY[l])::float / array_length(p_locations, 1)
           ELSE 0.0 
      END +
      CASE WHEN p_industries IS NOT NULL AND array_length(p_industries, 1) > 0 
           THEN (SELECT COUNT(*) FROM unnest(p_industries) i WHERE t.indexed_industries && ARRAY[i])::float / array_length(p_industries, 1)
           ELSE 0.0 
      END
    ) / GREATEST(
      (CASE WHEN p_keywords IS NOT NULL AND array_length(p_keywords, 1) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN p_locations IS NOT NULL AND array_length(p_locations, 1) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN p_industries IS NOT NULL AND array_length(p_industries, 1) > 0 THEN 1 ELSE 0 END),
      1
    ) as match_score
  FROM public.tenders t
  WHERE 
    t.is_duplicate = FALSE
    AND t.published_date >= v_cutoff
    AND (t.deadline IS NULL OR t.deadline >= v_now)
    -- Match ANY of the criteria (OR logic)
    AND (
      (p_keywords IS NOT NULL AND array_length(p_keywords, 1) > 0 AND t.indexed_keywords && p_keywords)
      OR (p_locations IS NOT NULL AND array_length(p_locations, 1) > 0 AND t.indexed_locations && p_locations)
      OR (p_industries IS NOT NULL AND array_length(p_industries, 1) > 0 AND t.indexed_industries && p_industries)
      OR (p_keywords IS NULL OR array_length(p_keywords, 1) = 0)
         AND (p_locations IS NULL OR array_length(p_locations, 1) = 0)
         AND (p_industries IS NULL OR array_length(p_industries, 1) = 0)
    )
  ORDER BY 
    match_score DESC,
    t.published_date DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$;

-- ============================================================================
-- RPC FUNCTION TO GET SEARCH TERM SUGGESTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION public.get_tender_search_suggestions(
  p_term_type text,
  p_limit integer DEFAULT 50
)
RETURNS TABLE (
  term_value text,
  term_display text,
  tender_count integer
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    ts.term_value,
    ts.term_display,
    ts.tender_count
  FROM public.tender_search_terms ts
  WHERE ts.term_type = p_term_type
    AND ts.tender_count > 0
  ORDER BY ts.tender_count DESC
  LIMIT p_limit;
END;
$$;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON public.tender_search_terms TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_tenders_indexed TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_tender_search_suggestions TO authenticated;


