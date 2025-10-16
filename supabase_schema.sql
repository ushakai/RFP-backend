-- Organizations / clients
create table if not exists public.clients (
  id uuid primary key default gen_random_uuid(),
  name text,
  sector text,
  contact_email text,
  api_key text unique,
  password_hash text,
  created_at timestamptz default now()
);

-- Questions per org
create table if not exists public.wifi_questions (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  original_text text not null,
  normalized_text text,
  embedding vector(1536),
  category text default 'Other',
  created_at timestamptz default now()
);

-- Answers per org
create table if not exists public.wifi_answers (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  answer_text text not null,
  answer_type text,
  character_count int,
  technical_level int,
  quality_score int,
  last_updated timestamptz default now()
);

-- Mappings
create table if not exists public.question_answer_mappings (
  id uuid primary key default gen_random_uuid(),
  question_id uuid references public.wifi_questions(id) on delete cascade,
  answer_id uuid references public.wifi_answers(id) on delete cascade,
  confidence_score float,
  context_requirements text,
  stakeholder_approved boolean default false,
  created_at timestamptz default now()
);

-- RPC for embedding match (pgvector required)
-- Ensure extension: create extension if not exists vector;
create or replace function public.match_wifi_questions(
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  p_client_id uuid
)
returns table(id uuid, question text, answer text, similarity float) language sql stable as $$
  select q.id, q.original_text as question, a.answer_text as answer,
         1 - (q.embedding <=> query_embedding) as similarity
  from public.wifi_questions q
  join public.question_answer_mappings m on m.question_id = q.id
  join public.wifi_answers a on a.id = m.answer_id
  where q.client_id = p_client_id
    and q.embedding is not null
    and (1 - (q.embedding <=> query_embedding)) >= match_threshold
  order by q.embedding <=> query_embedding
  limit match_count;
$$;
