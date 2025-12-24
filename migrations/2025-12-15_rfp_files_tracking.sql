-- Add file tracking fields to client_rfps table for reprocessing support
-- This allows storing original files and tracking last job info

ALTER TABLE public.client_rfps 
ADD COLUMN IF NOT EXISTS original_file_data bytea,
ADD COLUMN IF NOT EXISTS original_file_name text,
ADD COLUMN IF NOT EXISTS original_file_size bigint,
ADD COLUMN IF NOT EXISTS last_job_id uuid REFERENCES public.client_jobs(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS last_job_type text,
ADD COLUMN IF NOT EXISTS last_job_status text,
ADD COLUMN IF NOT EXISTS last_processed_at timestamptz;

-- Add index for querying RFPs by last job status
CREATE INDEX IF NOT EXISTS idx_client_rfps_last_job_status ON public.client_rfps(last_job_status);
CREATE INDEX IF NOT EXISTS idx_client_rfps_client_id_updated ON public.client_rfps(client_id, updated_at DESC);

-- Add comment for documentation
COMMENT ON COLUMN public.client_rfps.original_file_data IS 'Stores the original uploaded file for reprocessing';
COMMENT ON COLUMN public.client_rfps.last_job_id IS 'References the most recent job for this RFP';
COMMENT ON COLUMN public.client_rfps.last_job_status IS 'Cached status from last job: pending, processing, completed, failed';

