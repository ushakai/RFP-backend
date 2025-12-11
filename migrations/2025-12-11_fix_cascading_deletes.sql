-- Fix cascading deletes for activity_events table
-- This allows deleting users even if they have activity events

-- Drop the existing foreign key constraint
ALTER TABLE public.activity_events 
DROP CONSTRAINT IF EXISTS activity_events_actor_client_id_fkey;

-- Add it back with ON DELETE CASCADE
ALTER TABLE public.activity_events
ADD CONSTRAINT activity_events_actor_client_id_fkey
FOREIGN KEY (actor_client_id) 
REFERENCES public.clients(id) 
ON DELETE CASCADE;

-- Verify all other tables have proper cascading deletes
-- (This is a safety check - the schema should already have these)

-- Ensure client_sessions has cascade
ALTER TABLE public.client_sessions 
DROP CONSTRAINT IF EXISTS client_sessions_client_id_fkey;

ALTER TABLE public.client_sessions
ADD CONSTRAINT client_sessions_client_id_fkey
FOREIGN KEY (client_id) 
REFERENCES public.clients(id) 
ON DELETE CASCADE;

-- Ensure tender_matches has cascade
ALTER TABLE public.tender_matches 
DROP CONSTRAINT IF EXISTS tender_matches_client_id_fkey;

ALTER TABLE public.tender_matches
ADD CONSTRAINT tender_matches_client_id_fkey
FOREIGN KEY (client_id) 
REFERENCES public.clients(id) 
ON DELETE CASCADE;

-- Ensure tender_notifications has cascade
ALTER TABLE public.tender_notifications 
DROP CONSTRAINT IF EXISTS tender_notifications_client_id_fkey;

ALTER TABLE public.tender_notifications
ADD CONSTRAINT tender_notifications_client_id_fkey
FOREIGN KEY (client_id) 
REFERENCES public.clients(id) 
ON DELETE CASCADE;

-- Ensure tender_access has cascade
ALTER TABLE public.tender_access 
DROP CONSTRAINT IF EXISTS tender_access_client_id_fkey;

ALTER TABLE public.tender_access
ADD CONSTRAINT tender_access_client_id_fkey
FOREIGN KEY (client_id) 
REFERENCES public.clients(id) 
ON DELETE CASCADE;

-- Ensure user_tender_keywords has cascade
ALTER TABLE public.user_tender_keywords 
DROP CONSTRAINT IF EXISTS user_tender_keywords_client_id_fkey;

ALTER TABLE public.user_tender_keywords
ADD CONSTRAINT user_tender_keywords_client_id_fkey
FOREIGN KEY (client_id) 
REFERENCES public.clients(id) 
ON DELETE CASCADE;

