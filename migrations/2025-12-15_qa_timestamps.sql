-- Add created_at/updated_at to QA tables and ensure updated_at auto-refreshes

-- client_answers: created_at + updated_at
ALTER TABLE public.client_answers
    ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now(),
    ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- Backfill existing rows
UPDATE public.client_answers
SET
    created_at = COALESCE(created_at, now()),
    updated_at = COALESCE(updated_at, last_updated, now()),
    last_updated = COALESCE(last_updated, updated_at);

-- client_questions: updated_at
ALTER TABLE public.client_questions
    ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

UPDATE public.client_questions
SET updated_at = COALESCE(updated_at, created_at, now());

-- Trigger function to maintain updated_at
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

-- Attach triggers
DROP TRIGGER IF EXISTS trg_client_answers_updated_at ON public.client_answers;
CREATE TRIGGER trg_client_answers_updated_at
BEFORE UPDATE ON public.client_answers
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

DROP TRIGGER IF EXISTS trg_client_questions_updated_at ON public.client_questions;
CREATE TRIGGER trg_client_questions_updated_at
BEFORE UPDATE ON public.client_questions
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

