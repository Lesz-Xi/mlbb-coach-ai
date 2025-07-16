##TODOS##
1. Setup
# 1 A. Install Redis locally (Homebrew example)
brew install redis
brew services start redis  # or `redis-server` in another terminal

# 1 B. Add Python deps
pip install rq redis
echo "rq\nredis" >> requirements.txt

2.Reshape the backend  ➜  web/app.py
# --- top of file ---------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from redis import Redis
from rq import Queue
import uuid, os
from pathlib import Path

app = FastAPI()
redis_conn = Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379)
jobq = Queue("analysis", connection=redis_conn)

UPLOAD_DIR = Path("/tmp/skillshift_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------------------

def heavy_analysis(file_path: str) -> dict:
    """
    **This is your existing YOLO + OCR pipeline**.
    It must accept a *file path*, not UploadFile, so the worker
    can call it outside the FastAPI process.
    """
    # ... your long-running logic ...
    return {"hero": "Chou", "kda": "5/2/8", "confidence": 0.87}

# ---------- New enqueue route -------------------------------------
@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...)):
    # 1. Persist upload
    job_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with saved_path.open("wb") as f:
        f.write(await file.read())

    # 2. Enqueue
    job = jobq.enqueue("web.app.heavy_analysis", str(saved_path))
    return {"job_id": job.id, "state": job.get_status()}

# ---------- New status route --------------------------------------
@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobq.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    if job.is_finished:
        return {"state": "finished", "result": job.result}
    if job.is_failed:
        return {"state": "failed", "error": str(job.exc_info)}
    return {"state": job.get_status()}
    
Keep /api/health exactly as before and delete or deprecate the old blocking /api/analyze route to stop accidents.

3. Dedicated worker  ➜  worker.py  (run in its own container/terminal)
from redis import Redis
from rq import Worker, Queue, Connection
from web.app import heavy_analysis   # pulls in the function & its imports

redis_conn = Redis(host="localhost", port=6379)

with Connection(redis_conn):
    worker = Worker(queues=["analysis"])
    worker.work(burst=False)   # stay alive

This process imports YOLO once and never quits, so CUDA memory stays warm.

4. Wire-up the frontend
    1.	Create job
    const res = await fetch("/api/jobs", { method: "POST", body: formData });
    const { job_id } = await res.json();
    2.	Poll status
    const poll = async () => {
  const s = await fetch(`/api/jobs/${job_id}`).then(r => r.json());
  if (s.state === "finished") showResults(s.result);
  else if (s.state === "failed") toast.error(s.error);
  else setTimeout(poll, 2500);  // still "queued" or "started"
};
poll();

5. Update start_all_services.sh
#!/usr/bin/env bash
# ─── backend API (no workers) ──────────────────────────────────────
gunicorn web.app:app -k uvicorn.workers.UvicornWorker --workers 4 &

# ─── rq worker (1 per GPU; omit --with-scheduler unless you need) ─
python worker.py &

# ─── frontend dev server / vite / next.js etc. ────────────────────
npm run dev &
wait
(Leave Redis to run as a system service or docker container.)

6. Smoke test
# 1. Start Redis, API, worker.
./start_all_services.sh

# 2. Upload a screenshot
curl -F file=@screenshot.png localhost:8000/api/jobs | jq
#   ⇒ {"job_id":"abcdef123...", "state":"queued"}

# 3. Poll until finished
curl localhost:8000/api/jobs/abcdef123... | jq
#   ⇒ {"state":"finished","result":{...}}