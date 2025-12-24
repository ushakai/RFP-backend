-- Migration: Add worker_pid column to track which worker owns a job
-- This prevents ghost workers from incorrectly marking jobs as failed

-- Add worker_pid column to client_jobs
ALTER TABLE public.client_jobs ADD COLUMN IF NOT EXISTS worker_pid text;

-- Add last_updated column if missing
ALTER TABLE public.client_jobs ADD COLUMN IF NOT EXISTS last_updated timestamptz DEFAULT now();

-- Create index for faster worker lookup
CREATE INDEX IF NOT EXISTS idx_client_jobs_worker_pid ON public.client_jobs(worker_pid) WHERE worker_pid IS NOT NULL;

-- Comment explaining the column's purpose
COMMENT ON COLUMN public.client_jobs.worker_pid IS 'PID of the worker process that claimed this job. Used to prevent ghost workers from incorrectly marking jobs as failed.';
