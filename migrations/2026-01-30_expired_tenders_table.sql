-- Migration: Create expired_tenders table and move expired tenders
-- Date: 2026-01-30
-- Description: Creates a separate table for expired tenders to improve search performance

-- ============================================================================
-- EXPIRED TENDERS TABLE
-- Stores tenders that have passed their deadline to improve active tender search
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.expired_tenders (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  original_tender_id uuid NOT NULL, -- Reference to original tender (if needed for history)
  source text NOT NULL,
  external_id text NOT NULL,
  title text NOT NULL,
  description text,
  summary text,
  full_data jsonb NOT NULL,
  metadata jsonb,
  deadline timestamptz NOT NULL,
  published_date timestamptz,
  value_amount numeric,
  value_currency text,
  location text,
  category text,
  sector text,
  indexed_keywords text[],
  indexed_locations text[],
  indexed_industries text[],
  search_text text,
  expired_at timestamptz DEFAULT now(), -- When the tender was moved to expired
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(source, external_id)
);

-- Indexes for expired tenders
CREATE INDEX IF NOT EXISTS idx_expired_tenders_deadline ON public.expired_tenders(deadline DESC);
CREATE INDEX IF NOT EXISTS idx_expired_tenders_expired_at ON public.expired_tenders(expired_at DESC);
CREATE INDEX IF NOT EXISTS idx_expired_tenders_source ON public.expired_tenders(source);
CREATE INDEX IF NOT EXISTS idx_expired_tenders_original_id ON public.expired_tenders(original_tender_id);

-- GIN indexes for search (if needed for historical searches)
CREATE INDEX IF NOT EXISTS idx_expired_tenders_keywords ON public.expired_tenders USING GIN(indexed_keywords);
CREATE INDEX IF NOT EXISTS idx_expired_tenders_locations ON public.expired_tenders USING GIN(indexed_locations);
CREATE INDEX IF NOT EXISTS idx_expired_tenders_industries ON public.expired_tenders USING GIN(indexed_industries);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_expired_tenders_search_text ON public.expired_tenders USING GIN(to_tsvector('english', COALESCE(search_text, '')));

-- ============================================================================
-- FUNCTION: Move expired tenders to expired_tenders table
-- ============================================================================

CREATE OR REPLACE FUNCTION public.move_expired_tenders()
RETURNS TABLE(moved_count integer) AS $$
DECLARE
  moved integer := 0;
  tender_record RECORD;
BEGIN
  -- Find all tenders with passed deadlines that aren't already in expired_tenders
  FOR tender_record IN
    SELECT t.*
    FROM public.tenders t
    WHERE t.deadline IS NOT NULL
      AND t.deadline < now()
      AND NOT EXISTS (
        SELECT 1 FROM public.expired_tenders et
        WHERE et.source = t.source AND et.external_id = t.external_id
      )
  LOOP
    -- Insert into expired_tenders
    INSERT INTO public.expired_tenders (
      original_tender_id,
      source,
      external_id,
      title,
      description,
      summary,
      full_data,
      metadata,
      deadline,
      published_date,
      value_amount,
      value_currency,
      location,
      category,
      sector,
      indexed_keywords,
      indexed_locations,
      indexed_industries,
      search_text,
      expired_at,
      created_at,
      updated_at
    )
    VALUES (
      tender_record.id,
      tender_record.source,
      tender_record.external_id,
      tender_record.title,
      tender_record.description,
      tender_record.summary,
      tender_record.full_data,
      tender_record.metadata,
      tender_record.deadline,
      tender_record.published_date,
      tender_record.value_amount,
      tender_record.value_currency,
      tender_record.location,
      tender_record.category,
      tender_record.sector,
      tender_record.indexed_keywords,
      tender_record.indexed_locations,
      tender_record.indexed_industries,
      tender_record.search_text,
      now(),
      tender_record.created_at,
      tender_record.updated_at
    )
    ON CONFLICT (source, external_id) DO NOTHING;
    
    -- Delete related matches first (to avoid orphaned references)
    DELETE FROM public.tender_matches WHERE tender_id = tender_record.id;
    
    -- Delete from active tenders table
    DELETE FROM public.tenders WHERE id = tender_record.id;
    
    moved := moved + 1;
  END LOOP;
  
  RETURN QUERY SELECT moved;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: Get count of expired tenders to move
-- ============================================================================

CREATE OR REPLACE FUNCTION public.count_expired_tenders()
RETURNS integer AS $$
BEGIN
  RETURN (
    SELECT COUNT(*)
    FROM public.tenders
    WHERE deadline IS NOT NULL
      AND deadline < now()
  );
END;
$$ LANGUAGE plpgsql;

