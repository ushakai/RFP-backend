"""
Job Service - Background job processing
"""
import io
import os
import time
import traceback
import tempfile
import pandas as pd
from datetime import datetime
from config.settings import get_supabase_client
from services.excel_service import process_excel_file_obj, estimate_minutes_from_chars
from services.gemini_service import extract_qa_pairs_from_sheet
from services.supabase_service import insert_qa_pair, fetch_question_answer_mappings


def update_job_progress(job_id: str, progress: int, current_step: str, result_data: dict = None):
    """Update job progress in database with retry logic
    
    IMPORTANT: This function does NOT raise exceptions on failure.
    It only logs errors. This prevents status update failures from
    causing successful jobs to be marked as failed.
    """
    import inspect
    caller = inspect.stack()[1]
    caller_info = f"{caller.filename}:{caller.lineno} ({caller.function})"
    
    max_retries = 5  
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: [PID {os.getpid()}] Updating job {job_id} - Progress: {progress}%, Step: {current_step}")
            print(f"DEBUG: [PID {os.getpid()}] Update source: {caller_info}")
            
            # If marking as failed, log the stack trace to seeing where it came from
            if progress == -1:
                print(f"DEBUG: [PID {os.getpid()}] FAILED status requested for job {job_id}. Stack trace:")
                traceback.print_stack()

            # Safety check: If we are trying to update progress (0-99), but it's already terminal, skip it.
            # This prevents ghost workers from "resurrecting" a job with progress pulses.
            if progress >= 0 and progress < 100:
                try:
                    supabase = get_supabase_client()
                    current_job = supabase.table("client_jobs").select("status").eq("id", job_id).execute()
                    if current_job.data:
                        current_status = current_job.data[0].get("status")
                        if current_status in ["completed", "failed", "cancelled"]:
                            print(f"DEBUG: [PID {os.getpid()}] Safety catch: Refusing to update progress to {progress}% because job {job_id} is already in terminal state: {current_status.upper()}.")
                            return
                except Exception as e:
                    print(f"WARNING: [PID {os.getpid()}] Terminal status safety check failed: {e}")

            # Safety check: If we are trying to mark as FAILED, verify this PID owns the job
            # This prevents ghost workers (old worker processes) from incorrectly marking jobs as failed
            if progress == -1:
                try:
                    supabase = get_supabase_client()
                    current_job = supabase.table("client_jobs").select("status, worker_pid").eq("id", job_id).execute()
                    if current_job.data:
                        job_data = current_job.data[0]
                        current_status = job_data.get("status")
                        owner_pid = job_data.get("worker_pid")
                        my_pid = str(os.getpid())
                        
                        # If already completed, don't mark as failed
                        if current_status == "completed":
                            print(f"DEBUG: [PID {my_pid}] Safety catch: Refusing to mark job {job_id} as failed because it is already COMPLETED.")
                            return
                        
                        # If another PID owns this job, don't let this PID mark it as failed
                        if owner_pid and owner_pid != my_pid:
                            print(f"CRITICAL: [PID {my_pid}] BLOCKED: Refusing to mark job {job_id} as failed - job is owned by PID {owner_pid}. This is likely a ghost worker!")
                            return
                except Exception as e:
                    print(f"WARNING: [PID {os.getpid()}] PID ownership check failed: {e}")

            # Include PID and caller in step for easier debugging
            step_with_pid = f"[PID {os.getpid()}] {current_step}"
            
            updates = {
                "progress_percent": progress,
                "current_step": step_with_pid,
                "last_updated": datetime.now().isoformat()
            }
            if result_data:
                updates["result_data"] = result_data
            if progress == 100:
                updates["status"] = "completed"
                updates["completed_at"] = datetime.now().isoformat()
            elif progress == -1: # Use -1 for explicit failure
                updates["status"] = "failed"
                updates["completed_at"] = datetime.now().isoformat()
            
            supabase = get_supabase_client()
            # USE FILTER TO ENSURE WE ONLY UPDATE IF NOT COMPLETED (if this is a progress update)
            query = supabase.table("client_jobs").update(updates).eq("id", job_id)
            
            # If it's a non-terminal update, add a condition to the update query itself for extra safety
            if progress >= 0 and progress < 100:
                query = query.neq("status", "completed").neq("status", "failed")
            
            res = query.execute()
            
            if not res.data and progress >= 0 and progress < 100:
                print(f"DEBUG: [PID {os.getpid()}] Update affected 0 rows for job {job_id} - likely already terminal.")
                return

            
            # Update RFP with job status
            if progress == 100 or progress == -1:
                try:
                    job_res = supabase.table("client_jobs").select("rfp_id, job_type").eq("id", job_id).limit(1).execute()
                    if job_res.data:
                        rfp_id = job_res.data[0].get("rfp_id")
                        if rfp_id:
                            supabase.table("client_rfps").update({
                                "last_job_status": "completed" if progress == 100 else "failed",
                                "last_processed_at": datetime.now().isoformat() if progress == 100 else None
                            }).eq("id", rfp_id).execute()
                            print(f"Updated RFP {rfp_id} status to {'completed' if progress == 100 else 'failed'}")
                except Exception as rfp_e:
                    print(f"Warning: Failed to update RFP status: {rfp_e}")

            
            return  # Success, exit retry loop
        except Exception as e:
            print(f"ERROR: Error updating job progress {job_id} (attempt {attempt + 1}/{max_retries}): {e}")
            traceback.print_exc()
            if attempt < max_retries - 1:
                # Exponential backoff with connection reinitialization
                time.sleep(0.5 * (attempt + 1))
                try:
                    from config.settings import reinitialize_supabase
                    reinitialize_supabase()
                except:
                    pass
            else:
                print(f"CRITICAL ERROR: Failed to update job progress for {job_id} after {max_retries} attempts.")
                # DO NOT raise exception - just log it
                # This prevents status update failures from marking successful jobs as failed


def process_rfp_background(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Background RFP processing function using the working code"""
    print(f"DEBUG: Background RFP processing started for job {job_id}")
    start_time = time.time()
    
    try:
        update_job_progress(job_id, 10, "Starting RFP processing: Loading file...")
        
        # Check file size to prevent memory issues
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit
            raise Exception(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 50MB")
        
        # Validate Excel file before processing
        from services.excel_service import validate_excel_file
        is_valid, error_msg = validate_excel_file(file_content, file_name)
        if not is_valid:
            raise ValueError(f"Invalid Excel file: {error_msg}")
        
        print(f"DEBUG: Processing file {file_name} ({file_size_mb:.1f}MB)")
        
        file_obj = io.BytesIO(file_content)
        processed_output, processed_sheets_count, total_questions_processed = process_excel_file_obj(
            file_obj, file_name, client_id, rfp_id, job_id=job_id, 
            update_progress_callback=update_job_progress
        )
        
        processed_content = processed_output.getvalue()
        
        update_job_progress(job_id, 95, "Finalizing and storing processed file...")
        
        # Store both original and processed files for comparison
        import base64
        processed_file_b64 = base64.b64encode(processed_content).decode('utf-8')
        original_file_b64 = base64.b64encode(file_content).decode('utf-8')
        
        result_data = {
            "file_name": f"processed_{file_name}",
            "file_size": len(processed_content),
            "processing_completed": True,
            "processing_time_seconds": int(time.time() - start_time),
            "processed_file": processed_file_b64,  # Store as base64
            "original_file": original_file_b64,    # Store original for comparison
            "sheets_processed": processed_sheets_count,
            "total_questions_processed": total_questions_processed
        }
        
        update_job_progress(job_id, 100, "RFP processing completed successfully!", result_data)
        print(f"DEBUG: Background RFP processing completed successfully for job {job_id} in {time.time() - start_time:.1f}s")
                
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Processing failed after {processing_time:.1f}s: {str(e)}"
        print(f"ERROR: RFP processing background error for job {job_id}: {error_msg}")
        traceback.print_exc()
        update_job_progress(job_id, -1, error_msg)


def extract_qa_background(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Background QA extraction function"""
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from config.settings import GEMINI_MODEL
    
    print(f"DEBUG: Background QA extraction started for job {job_id}")
    try:
        update_job_progress(job_id, 10, "Starting QA extraction: Loading file...")
        
        # Validate file before processing
        from services.excel_service import validate_excel_file
        is_valid, error_msg = validate_excel_file(file_content, file_name)
        if not is_valid:
            raise ValueError(f"Invalid Excel file: {error_msg}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            update_job_progress(job_id, 20, "Loading Excel file for QA extraction...")
            from services.excel_service import read_excel_with_fallback
            xls = read_excel_with_fallback(tmp_path)
            
            total_sheets = len(xls.sheet_names)
            processed_sheets = 0
            extracted_pairs = []
            
            for sheet_idx, sheet in enumerate(xls.sheet_names):
                progress_start_sheet = 20 + (sheet_idx * 70 // total_sheets)
                
                update_job_progress(job_id, progress_start_sheet, f"Extracting from sheet {sheet_idx + 1}/{total_sheets}: {sheet}")
                
                # Try to read with openpyxl first, fallback to xlrd if needed
                try:
                    df = pd.read_excel(tmp_path, sheet_name=sheet, header=None, engine="openpyxl")
                except Exception:
                    # Fallback to xlrd for older .xls files
                    df = pd.read_excel(tmp_path, sheet_name=sheet, header=None, engine="xlrd")
                if df.empty:
                    print(f"DEBUG: Sheet {sheet} is empty, skipping for QA extraction.")
                    processed_sheets += 1
                    continue
                
                sheet_csv = df.to_csv(index=False, header=False)
                pairs = extract_qa_pairs_from_sheet(sheet_csv)
                # Retry up to 2 additional times if no pairs extracted
                retry_attempt = 0
                while (not pairs or len(pairs) == 0) and retry_attempt < 2:
                    retry_attempt += 1
                    print(f"DEBUG: No Q&A pairs extracted from sheet {sheet}. Retrying attempt {retry_attempt}/2...")
                    pairs = extract_qa_pairs_from_sheet(sheet_csv)
                
                for p in pairs:
                    extracted_pairs.append({
                        "question": p.get("question", ""),
                        "answer": p.get("answer", ""),
                        "category": p.get("category", "Other"),
                        "sheet": sheet
                    })
                
                processed_sheets += 1
            
            update_job_progress(job_id, 90, f"Saving {len(extracted_pairs)} extracted Q&A pairs to database...")
            print(f"DEBUG: Saving {len(extracted_pairs)} extracted Q&A pairs to database for job {job_id}.")
            
            created_count = 0
            for p in extracted_pairs:
                q = p.get("question", "").strip()
                a = p.get("answer", "").strip()
                c = p.get("category", "Other").strip() or "Other"
                if q and a and insert_qa_pair(client_id, q, a, c, rfp_id):
                    created_count += 1
            
            # Build AI groups after extraction
            ai_groups_result = _build_ai_groups_for_job(client_id, rfp_id)

            # Save pending summaries
            try:
                supabase = get_supabase_client()
                for grp in ai_groups_result.get("groups", []):
                    cq = (grp.get("consolidated_question") or "").strip()
                    ca = (grp.get("consolidated_answer") or "").strip()
                    qids = grp.get("question_ids") or []
                    if not cq or not ca or not qids:
                        continue
                    s_ins = supabase.table("client_summaries").insert({
                        "summary_text": ca,
                        "summary_type": "Consolidated",
                        "character_count": len(ca),
                        "quality_score": None,
                        "approved": False,
                        "client_id": client_id,
                        "rfp_id": rfp_id,
                    }).execute()
                    s_id = (s_ins.data or [{}])[0].get("id")
                    if s_id:
                        mappings = [{"question_id": qid, "summary_id": s_id} for qid in qids]
                        supabase.table("client_question_summary_mappings").insert(mappings).execute()
            except Exception as e:
                print(f"WARN: Failed to persist pending summaries: {e}")
                traceback.print_exc()

            result_data = {
                "extracted_pairs_count": len(extracted_pairs), 
                "created_pairs_count": created_count,          
                "total_sheets_processed": processed_sheets,
                "ai_groups": ai_groups_result.get("groups", []),
                "ai_groups_count": len(ai_groups_result.get("groups", [])),
            }
            
            update_job_progress(job_id, 100, "QA extraction completed successfully!", result_data)
            print(f"DEBUG: Background QA extraction completed successfully for job {job_id}")
            
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    print(f"DEBUG: Cleaned up temporary file: {tmp_path}")
                except PermissionError:
                    print(f"DEBUG: Could not delete temp file {tmp_path} - file may be in use")
                except Exception as e:
                    print(f"DEBUG: Error cleaning up temp file: {e}")
                    traceback.print_exc()
                
    except Exception as e:
        print(f"ERROR: QA extraction background error for job {job_id}: {e}")
        traceback.print_exc()
        update_job_progress(job_id, -1, f"Extraction failed: {str(e)}")


def _build_ai_groups_for_job(client_id: str, rfp_id: str | None):
    """Build AI-driven Q&A groups for a job"""
    import json
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from config.settings import GEMINI_MODEL
    
    try:
        supabase = get_supabase_client()
        q_query = supabase.table("client_questions").select("id, original_text, rfp_id").eq("client_id", client_id)
        if rfp_id:
            q_query = q_query.eq("rfp_id", rfp_id)
        questions_local = q_query.order("created_at", desc=True).execute().data or []

        if not questions_local:
            return {"groups": [], "message": "No questions found"}

        q_ids_local = [q["id"] for q in questions_local]
        m_rows_local = fetch_question_answer_mappings(q_ids_local)
        q_to_a_local = {m["question_id"]: m["answer_id"] for m in m_rows_local}

        a_ids_local = list({aid for aid in q_to_a_local.values() if aid})
        a_rows_local = []
        if a_ids_local:
            a_rows_local = supabase.table("client_answers").select("id, answer_text").in_("id", a_ids_local).execute().data or []
        a_map_local = {a["id"]: a.get("answer_text", "") for a in a_rows_local}

        qa_lines_local = []
        for q in questions_local:
            qid = q["id"]
            qtext = (q.get("original_text") or "").strip()
            atext = (a_map_local.get(q_to_a_local.get(qid)) or "").strip()
            if not qtext:
                continue
            qa_lines_local.append({"id": qid, "q": qtext, "a": atext})

        if not qa_lines_local:
            return {"groups": [], "message": "No Q&A pairs available"}

        qa_text_local = "\n".join([f"ID:{row['id']}\nQ:{row['q']}\nA:{row['a']}" for row in qa_lines_local])
        prompt_local = f"""
You will receive a list of Q&A pairs for a single client (and optionally a specific RFP). Group semantically similar questions together and produce a consolidated Q&A for each group.

Return ONLY strict JSON with this structure:
{{
  "groups": [
    {{
      "question_ids": ["<id>", "<id>", ...],
      "consolidated_question": "string",
      "consolidated_answer": "string"
    }}
  ]
}}

Rules:
1) "question_ids" must be the exact IDs provided.
2) Every ID used must exist in the input. Do not invent IDs.
3) Do not include any text before or after the JSON.

Q&A LIST:
{qa_text_local}
"""
        try:
            gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            response_local = gemini_model.generate_content(
                prompt_local,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                },
            )
            text_local = (response_local.text or "").strip()
            s_local, e_local = text_local.find("{"), text_local.rfind("}")
            if s_local == -1 or e_local == -1:
                return {"groups": [], "message": "LLM returned no JSON"}
            data_local = json.loads(text_local[s_local:e_local+1])

            raw_groups_local = data_local.get("groups") or []
            valid_ids_local = {str(x["id"]) for x in questions_local}
            groups_local = []
            for g in raw_groups_local:
                qids_local = [str(x) for x in (g.get("question_ids") or []) if str(x) in valid_ids_local]
                if not qids_local:
                    continue
                groups_local.append({
                    "question_ids": qids_local,
                    "consolidated_question": (g.get("consolidated_question") or "").strip(),
                    "consolidated_answer": (g.get("consolidated_answer") or "").strip(),
                })
            return {"groups": groups_local}
        except Exception as e:
            print(f"_build_ai_groups_for_job LLM error: {e}")
            traceback.print_exc()
            return {"groups": [], "message": "AI grouping failed"}
    except Exception as e:
        print(f"_build_ai_groups_for_job error: {e}")
        traceback.print_exc()
        return {"groups": [], "message": "AI grouping error"}


def ingest_text_background(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Background text document ingestion function"""
    print(f"DEBUG: Background text ingestion started for job {job_id}")
    start_time = time.time()
    
    try:
        # Initial progress update (marks as processing via status check and updates step)
        update_job_progress(job_id, 10, "Starting text ingestion: Loading file...")
        
        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > 50:
            raise Exception(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 50MB")
        
        print(f"DEBUG: Processing text file {file_name} ({file_size_mb:.1f}MB)")
        
        update_job_progress(job_id, 20, "Extracting text from document...")
        
        # Use the new production-grade ingestion pipeline
        from services.document_ingestion_service import DocumentIngestionPipeline
        
        # Detect file type from filename
        import os
        ext = os.path.splitext(file_name)[1].lower()
        mime_map = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.markdown': 'text/markdown',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        }
        content_type = mime_map.get(ext, 'text/plain')
        
        update_job_progress(job_id, 30, "Initializing ingestion pipeline...")
        
        # Initialize ingestion pipeline
        pipeline = DocumentIngestionPipeline(
            client_id=client_id,
            rfp_id=rfp_id,
        )
        
        update_job_progress(job_id, 40, "Running ingestion pipeline (extraction, chunking, embedding)...")
        
        # Run ingestion
        result = pipeline.ingest_file(
            file_content=file_content,
            filename=file_name,
            content_type=content_type,
            metadata={
                "source": "background_job",
                "job_id": job_id,
            }
        )
        
        print(f"DEBUG: Ingestion pipeline result: success={result.get('success')}")
        if not result.get('success'):
            print(f"DEBUG: Result details: {result}")
        
        if not result['success']:
            error_msg = result.get('error', 'Unknown error during ingestion')
            print(f"ERROR: Ingestion pipeline failed: {error_msg}")
            raise Exception(error_msg)
        
        chunks_stored = result['chunks_stored']
        chunks_failed = result['chunks_failed']
        quality_score = result['quality_score']
        
        print(f"DEBUG: Chunks stored: {chunks_stored}, failed: {chunks_failed}, quality: {quality_score:.2f}")
        
        # Check if we actually stored any chunks
        if chunks_stored == 0:
            raise Exception(
                f"No chunks were stored. This can happen if:\n"
                f"1. The document is too short (min {50} tokens required per chunk)\n"
                f"2. Text extraction failed\n"
                f"3. Quality score too low ({quality_score:.2f})\n"
                f"Failed chunks: {chunks_failed}"
            )
        
        update_job_progress(job_id, 90, f"Finalizing... Stored {chunks_stored} chunks")
        
        # Prepare result data
        result_data = {
            "chunks_stored": chunks_stored,
            "chunks_failed": chunks_failed,
            "quality_score": quality_score,
            "extraction_time": result['statistics']['extraction_time'],
            "chunking_time": result['statistics']['chunking_time'],
            "embedding_time": result['statistics']['embedding_time'],
            "storage_time": result['statistics']['storage_time'],
            "total_time": result['statistics']['total_time'],
            "chunks_deduplicated": result['statistics'].get('chunks_deduplicated', 0),
        }
        
        # Log activity
        try:
            from services.activity_service import record_event
            record_event(
                "qa",
                "text_ingested",
                actor_client_id=client_id,
                subject_type="doc_batch",
                subject_id=rfp_id,
                metadata={
                    "job_id": job_id,
                    "file_name": file_name,
                    "chunks_stored": chunks_stored,
                    "quality_score": quality_score,
                },
            )
        except Exception as e:
            print(f"Warning: Failed to log activity event: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"DEBUG: Text ingestion completed for job {job_id} in {elapsed_time:.2f}s")
        print(f"DEBUG: Stored {chunks_stored} chunks, Quality score: {quality_score:.2f}")
        
        # Final update to mark as completed - CRITICAL: Don't let status update failures fail the job
        # The ingestion already succeeded, so we must mark it as completed
        # Wrap in try-except to ensure we never raise exceptions that would mark the job as failed
        try:
            max_status_update_attempts = 5
            status_updated = False
            
            for attempt in range(max_status_update_attempts):
                try:
                    # update_job_progress doesn't raise exceptions, but we catch just in case
                    update_job_progress(
                        job_id, 
                        100, 
                        f"Completed: {chunks_stored} chunks ingested",
                        result_data
                    )
                    print(f"DEBUG: Job {job_id} marked as completed")
                    status_updated = True
                    break
                except Exception as update_error:
                    print(f"ERROR: update_job_progress raised exception (attempt {attempt + 1}/{max_status_update_attempts}): {update_error}")
                    if attempt < max_status_update_attempts - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        # Reinitialize connection
                        try:
                            from config.settings import reinitialize_supabase
                            reinitialize_supabase()
                        except:
                            pass
            
            # If update_job_progress failed, try direct database update
            if not status_updated:
                print(f"WARNING: update_job_progress failed, trying direct database update")
                for attempt in range(3):
                    try:
                        supabase_client = get_supabase_client()
                        supabase_client.table("client_jobs").update({
                            "status": "completed",
                            "progress_percent": 100,
                            "current_step": f"Completed: {chunks_stored} chunks ingested",
                            "completed_at": datetime.now().isoformat(),
                            "result_data": result_data,
                            "last_updated": datetime.now().isoformat()
                        }).eq("id", job_id).execute()
                        
                        # Also update RFP status
                        try:
                            supabase_client.table("client_rfps").update({
                                "last_job_status": "completed",
                                "last_processed_at": datetime.now().isoformat()
                            }).eq("id", rfp_id).execute()
                        except:
                            pass
                        
                        print(f"DEBUG: Job {job_id} marked as completed (direct update)")
                        status_updated = True
                        break
                    except Exception as retry_error:
                        print(f"ERROR: Direct status update failed (attempt {attempt + 1}/3): {retry_error}")
                        if attempt < 2:
                            time.sleep(0.5 * (attempt + 1))
                            try:
                                from config.settings import reinitialize_supabase
                                reinitialize_supabase()
                            except:
                                pass
            
            # If we still couldn't update status, log it but don't fail the job
            # The ingestion succeeded, so we just log the status update failure
            if not status_updated:
                print(f"CRITICAL: Cannot mark job {job_id} as completed after all attempts, but ingestion succeeded!")
                print(f"CRITICAL: Job processed successfully - {chunks_stored} chunks stored")
                # Try one final time with a simple status update
                try:
                    supabase_client = get_supabase_client()
                    supabase_client.table("client_jobs").update({
                        "status": "completed",
                        "progress_percent": 100
                    }).eq("id", job_id).execute()
                    print(f"DEBUG: Job {job_id} marked as completed (final simple update)")
                except Exception as final_error:
                    print(f"CRITICAL: Even final simple status update failed: {final_error}")
        except Exception as status_error:
            # CRITICAL: Never let status update failures cause the job to be marked as failed
            # The ingestion succeeded, so we must not raise exceptions
            print(f"CRITICAL: Unexpected error in status update (but ingestion succeeded): {status_error}")
            traceback.print_exc()
            # Don't re-raise - job succeeded, status update just failed
        
    except Exception as e:
        error_message = f"Text ingestion failed: {str(e)}"
        print(f"ERROR: {error_message}")
        traceback.print_exc()
        
        try:
            update_job_progress(
                job_id, 
                -1, 
                f"Ingestion failed: {str(e)}",
                {"error": str(e)}
            )
        except Exception as update_error:
            print(f"ERROR: Failed to update job failure status: {update_error}")

