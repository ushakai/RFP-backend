-- Admin dashboard schema updates
-- Safe to run multiple times (idempotent where possible)

-- 1) Extend clients with admin-related fields
alter table if exists public.clients
  add column if not exists role text default 'client',
  add column if not exists status text default 'active',
  add column if not exists api_key_revoked boolean default false,
  add column if not exists last_login_at timestamptz,
  add column if not exists last_active_at timestamptz;

create index if not exists idx_clients_status on public.clients(status);
create index if not exists idx_clients_role on public.clients(role);

-- 2) Activity events table (audit log)
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

-- 3) Client sessions table (for admin revocation)
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

-- 4) Backfill helpers (no-op if data already present)
update public.clients
set role = coalesce(nullif(role, ''), 'client')
where role is null or role = '';

update public.clients
set status = coalesce(nullif(status, ''), 'active')
where status is null or status = '';

-- 5) Optional: mark known admins by email (set ADMIN_EMAILS env accordingly before running)
-- Example:
-- update public.clients set role = 'admin'
-- where lower(contact_email) in ('admin@rfp.com');


