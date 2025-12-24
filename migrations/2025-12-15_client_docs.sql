-- Client documents table for text ingestion and RAG
-- Stores chunked text with embeddings

create table if not exists public.client_docs (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  rfp_id uuid references public.client_rfps(id) on delete cascade,
  filename text,
  mime_type text,
  size_bytes bigint,
  chunk_id text,
  chunk_index int,
  content_text text,
  embedding vector(768),
  metadata jsonb,
  created_at timestamptz default now()
);

create index if not exists idx_client_docs_client on public.client_docs(client_id, created_at desc);
create index if not exists idx_client_docs_rfp on public.client_docs(rfp_id, created_at desc);
create index if not exists idx_client_docs_chunk on public.client_docs(chunk_id);

-- Optional: vector index (works if pgvector index is available)
-- create index if not exists idx_client_docs_embedding on public.client_docs using ivfflat (embedding vector_cosine_ops) with (lists = 100);

