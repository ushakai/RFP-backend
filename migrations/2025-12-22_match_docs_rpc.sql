-- Migration: Update match_client_docs to include original_rfp_date
-- This enables date-based prioritization for document chunks

create or replace function public.match_client_docs(
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  p_client_id uuid,
  p_rfp_id uuid default null
)
returns table(
  id uuid,
  content_text text,
  filename text,
  chunk_index int,
  metadata jsonb,
  created_at timestamptz,
  similarity float,
  original_rfp_date date
) language sql stable as $$
  select
    d.id,
    d.content_text,
    d.filename,
    d.chunk_index,
    d.metadata,
    d.created_at,
    1 - (d.embedding <=> query_embedding) as similarity,
    r.original_rfp_date
  from public.client_docs d
  left join public.client_rfps r on d.rfp_id = r.id
  where d.client_id = p_client_id
    and (p_rfp_id is null or d.rfp_id = p_rfp_id)
    and d.embedding is not null
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;
