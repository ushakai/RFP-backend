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

-- RFPs per client
create table if not exists public.client_rfps (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  name text not null,
  description text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.client_questions (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  rfp_id uuid references public.client_rfps(id) on delete cascade,
  original_text text not null,
  normalized_text text,
  embedding vector(768),
  category text default 'Other',
  created_at timestamptz default now()
);

create table if not exists public.client_answers (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  rfp_id uuid references public.client_rfps(id) on delete cascade,
  answer_text text not null,
  answer_type text,
  character_count int,
  technical_level int,
  quality_score int,
  last_updated timestamptz default now()
);

-- Mappings (renamed to client_question_answer_mappings)
create table if not exists public.client_question_answer_mappings (
  id uuid primary key default gen_random_uuid(),
  question_id uuid references public.client_questions(id) on delete cascade,
  answer_id uuid references public.client_answers(id) on delete cascade,
  confidence_score float,
  context_requirements text,
  stakeholder_approved boolean default false,
  created_at timestamptz default now()
);

-- Job processing table
create table if not exists public.client_jobs (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  rfp_id uuid references public.client_rfps(id) on delete cascade,
  job_type text not null, -- 'process_rfp' or 'extract_qa'
  status text default 'pending', -- 'pending', 'processing', 'completed', 'failed'
  file_name text,
  file_size bigint,
  progress_percent int default 0,
  current_step text,
  estimated_completion timestamptz,
  started_at timestamptz,
  completed_at timestamptz,
  error_message text,
  job_data jsonb, -- stores input file content and parameters
  result_data jsonb, -- stores extracted questions/answers or processing results
  created_at timestamptz default now()
);

-- Ensure extension: create extension if not exists vector;
create or replace function public.client_match_questions(
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  p_client_id uuid,
  p_rfp_id uuid default null
)
returns table(id uuid, question text, answer text, similarity float) language sql stable as $$
  select q.id, q.original_text as question, a.answer_text as answer,
         1 - (q.embedding <=> query_embedding) as similarity
  from public.client_questions q
  join public.client_question_answer_mappings m on m.question_id = q.id
  join public.client_answers a on a.id = m.answer_id
  where q.client_id = p_client_id
    and (p_rfp_id is null or q.rfp_id = p_rfp_id)
    and q.embedding is not null
    and (1 - (q.embedding <=> query_embedding)) >= match_threshold
  order by q.embedding <=> query_embedding
  limit match_count;
$$;

-- Function to add a new Q&A pair
create or replace function public.add_qa_pair(
  p_client_key text,
  p_question text,
  p_answer text,
  p_category text default 'General',
  p_rfp_id uuid default null
) returns void language plpgsql security definer as $$
declare
  v_client_id uuid;
  v_question_id uuid;
  v_answer_id uuid;
begin
  -- Get client ID from key
  select id into v_client_id from public.clients where api_key = p_client_key;
  if not found then
    raise exception 'Invalid client key';
  end if;

  -- Insert question
  insert into public.client_questions (
    client_id, rfp_id, original_text, normalized_text, category
  ) values (
    v_client_id, p_rfp_id, p_question, lower(p_question), p_category
  ) returning id into v_question_id;

  -- Insert answer
  insert into public.client_answers (
    client_id, rfp_id, answer_text, answer_type, character_count, technical_level
  ) values (
    v_client_id, p_rfp_id, p_answer, 'General', length(p_answer), 1
  ) returning id into v_answer_id;

  -- Create mapping
  insert into public.client_question_answer_mappings (
    question_id, answer_id, confidence_score, stakeholder_approved
  ) values (
    v_question_id, v_answer_id, 1.0, false
  );
end;
$$;
