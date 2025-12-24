-- Add OTP verification columns to clients table
ALTER TABLE public.clients 
ADD COLUMN IF NOT EXISTS otp_code TEXT,
ADD COLUMN IF NOT EXISTS otp_expires_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS otp_attempts INT DEFAULT 0;

-- Update existing clients to active if they don't have a status
UPDATE public.clients SET status = 'active' WHERE status IS NULL;

-- Ensure the status column has a default of 'pending_verification' for new users
-- Note: We'll handle setting the status in the backend code during registration.
