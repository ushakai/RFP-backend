-- Organizations / clients
create table if not exists public.clients (
  id uuid primary key default gen_random_uuid(),
  name text,
  sector text,
  contact_email text,
  api_key text unique,
  password_hash text,
  role text default 'client',
  status text default 'active',
  api_key_revoked boolean default false,
  last_login_at timestamptz,
  last_active_at timestamptz,
  created_at timestamptz default now()
);

-- Indexes for admin queries
create index if not exists idx_clients_status on public.clients(status);
create index if not exists idx_clients_role on public.clients(role);

-- Tender ingestion log (tracks daily ingestions)
create table if not exists public.tender_ingestion_log (
  id uuid primary key default gen_random_uuid(),
  ingested_at timestamptz default now(),
  ingested_date date unique
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

-- Summaries per client (consolidated across similar questions)
create table if not exists public.client_summaries (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  rfp_id uuid references public.client_rfps(id) on delete cascade,
  summary_text text not null,
  summary_type text, -- e.g., 'Consolidated'
  character_count int,
  quality_score int,
  approved boolean default false,
  created_at timestamptz default now()
);

-- Mapping questions to a summary
create table if not exists public.client_question_summary_mappings (
  id uuid primary key default gen_random_uuid(),
  question_id uuid references public.client_questions(id) on delete cascade,
  summary_id uuid references public.client_summaries(id) on delete cascade,
  created_at timestamptz default now()
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

-- ============================================================================
-- TENDER MONITORING SYSTEM TABLES
-- ============================================================================

-- Tenders table - stores all tender opportunities from various APIs
create table if not exists public.tenders (
  id uuid primary key default gen_random_uuid(),
  source text not null, -- 'TED', 'FindATender', 'ContractsFinder', 'PCS', 'Sell2Wales', 'SAM', 'AusTender'
  external_id text not null, -- ID from the source API
  title text not null,
  description text,
  summary text, -- anonymised summary for notifications
  full_data jsonb not null, -- complete tender data from API
  metadata jsonb, -- generated metadata (categories, tags, etc.)
  deadline timestamptz,
  published_date timestamptz,
  value_amount numeric,
  value_currency text,
  location text,
  category text,
  sector text,
  is_duplicate boolean default false,
  duplicate_of uuid references public.tenders(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(source, external_id)
);

-- User tender keywords - criteria for matching tenders
create table if not exists public.user_tender_keywords (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  keywords text[] not null, -- array of keywords to match
  match_type text default 'any', -- 'any' (OR) or 'all' (AND)
  categories text[], -- optional category filters
  sectors text[], -- optional sector filters
  min_value numeric, -- optional minimum value filter
  max_value numeric, -- optional maximum value filter
  locations text[], -- optional location filters
  is_active boolean default true,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- ============================================================================
-- ADMIN AUDIT & SESSIONS
-- ============================================================================
create table if not exists public.activity_events (
  id uuid primary key default gen_random_uuid(),
  event_type text not null, -- auth, bid, file, system
  action text not null,
  actor_client_id uuid references public.clients(id),
  actor_email text,
  subject_id text,
  subject_type text,
  metadata jsonb,
  created_at timestamptz default now()
);
create index if not exists idx_activity_events_created_at on public.activity_events(created_at desc);
create index if not exists idx_activity_events_actor on public.activity_events(actor_client_id);
create index if not exists idx_activity_events_type on public.activity_events(event_type);

create table if not exists public.client_sessions (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  api_key text unique,
  user_agent text,
  ip_address text,
  created_at timestamptz default now(),
  last_seen_at timestamptz,
  revoked boolean default false,
  revoked_by text,
  revoked_at timestamptz
);
create index if not exists idx_client_sessions_client on public.client_sessions(client_id);
create index if not exists idx_client_sessions_last_seen on public.client_sessions(last_seen_at desc);

-- Tender matches - links tenders to users based on keyword matching
create table if not exists public.tender_matches (
  id uuid primary key default gen_random_uuid(),
  tender_id uuid references public.tenders(id) on delete cascade,
  client_id uuid references public.clients(id) on delete cascade,
  keyword_set_id uuid references public.user_tender_keywords(id) on delete cascade,
  match_score float, -- relevance score (0-1)
  matched_keywords text[], -- which keywords matched
  created_at timestamptz default now(),
  unique(tender_id, client_id, keyword_set_id)
);

-- Tender notifications - tracks email notifications sent to users
create table if not exists public.tender_notifications (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  tender_id uuid references public.tenders(id) on delete cascade,
  match_id uuid references public.tender_matches(id) on delete cascade,
  notification_type text default 'daily_digest', -- 'daily_digest', 'individual'
  email_sent boolean default false,
  email_sent_at timestamptz,
  email_subject text,
  created_at timestamptz default now()
);

-- Tender access - tracks which users have paid for full access to tenders
create table if not exists public.tender_access (
  id uuid primary key default gen_random_uuid(),
  client_id uuid references public.clients(id) on delete cascade,
  tender_id uuid references public.tenders(id) on delete cascade,
  payment_status text default 'pending', -- 'pending', 'completed', 'failed'
  payment_amount numeric default 5.00, -- £5 per tender
  payment_reference text,
  payment_date timestamptz,
  access_granted_at timestamptz,
  created_at timestamptz default now(),
  unique(client_id, tender_id)
);

-- Indexes for performance
create index if not exists idx_tenders_source on public.tenders(source);
create index if not exists idx_tenders_published_date on public.tenders(published_date);
create index if not exists idx_tenders_deadline on public.tenders(deadline);
create index if not exists idx_tenders_category on public.tenders(category);
create index if not exists idx_tender_matches_client on public.tender_matches(client_id);
create index if not exists idx_tender_matches_tender on public.tender_matches(tender_id);
create index if not exists idx_tender_notifications_client on public.tender_notifications(client_id);
create index if not exists idx_tender_access_client on public.tender_access(client_id);
create index if not exists idx_tender_access_tender on public.tender_access(tender_id);
create index if not exists idx_user_tender_keywords_client on public.user_tender_keywords(client_id);
