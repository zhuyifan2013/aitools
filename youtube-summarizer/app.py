from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, AnyUrl
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any, Literal
import uuid
import time
import threading
import traceback
import subprocess
import os
import tempfile
import math

app = FastAPI()

API_TOKEN = "yifan_token"  # 你可以改更长更随机

def check_auth(authorization: Optional[str]):
    if authorization not in (f"Bearer {API_TOKEN}", API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---- Models ----
JobStatus = Literal["queued", "running", "done", "failed"]

class YouTubeJobCreateReq(BaseModel):
    youtube_url: AnyUrl

class YouTubeJobCreateResp(BaseModel):
    job_id: str
    status: JobStatus

class YouTubeJobResp(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    error: Optional[str] = None

class YouTubeJobResultResp(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ---- In-memory store (先用内存，后面可换 sqlite/redis) ----
JOBS: Dict[str, Dict[str, Any]] = {}

def _now() -> float:
    return time.time()

WHISPER_MODEL = "base"  # 按你要求
_whisper_model = None  # 进程级缓存

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",          # 你如果有 GPU 以后再改
            compute_type="int8"     # base + int8 非常快
        )
    return _whisper_model


def _run_youtube_pipeline(job_id: str):
    try:
        job = JOBS[job_id]
        job["status"] = "running"
        job["updated_at"] = _now()
        print("[job]", job_id, "start")
        youtube_url = job["youtube_url"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1️⃣ 下载音频（m4a）
            output_tpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

            print("[job]", tmpdir, " tmpdir")
            cmd = [
                "yt-dlp",
                "--cookies-from-browser", "chrome",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "m4a",
                "--audio-quality", "5",
                "-o", output_tpl,
                youtube_url,
            ]

            subprocess.run(cmd, check=True)
            print("[job] yt-dlp downloading")
            # 找到下载好的 m4a
            audio_file = None
            for f in os.listdir(tmpdir):
                if f.endswith(".m4a"):
                    audio_file = os.path.join(tmpdir, f)
                    break

            if not audio_file:
                raise RuntimeError("yt-dlp did not produce m4a file")

            # 2️⃣ Whisper 转写
            model = get_whisper_model()

            segments, info = model.transcribe(
                audio_file,
                beam_size=5,
                vad_filter=True
            )

            print("[job] whisper transcribing")
            segs = list(segments)  # ✅ generator -> list，后面才能索引/取最后一个

            paragraphs = []
            buffer = []
            start_ts = None
            max_duration = 150.0

            for seg in segs:
                if start_ts is None:
                    start_ts = seg.start

                buffer.append(seg.text.strip())

                if seg.end - start_ts >= max_duration:
                    paragraphs.append({
                        "index": len(paragraphs) + 1,
                        "start": round(start_ts, 2),
                        "end": round(seg.end, 2),
                        "text": " ".join(buffer)
                    })
                    buffer = []
                    start_ts = None

            # 收尾（注意 segs 可能为空）
            if buffer:
                end_ts = segs[-1].end if segs else (start_ts or 0.0)
                paragraphs.append({
                    "index": len(paragraphs) + 1,
                    "start": round(start_ts or 0.0, 2),
                    "end": round(end_ts, 2),
                    "text": " ".join(buffer)
                })

            print("[job] paragraphs:", len(paragraphs))
            full_text = "\n\n".join(p["text"] for p in paragraphs)

            # 4️⃣ 写入结果（给 GPT 用）
            job["result"] = {
                "language": info.language,
                "paragraphs": paragraphs,
                "full_text": full_text,
            }

            job["status"] = "done"
            job["updated_at"] = _now()

    except Exception as e:
        job = JOBS.get(job_id, {})
        job["status"] = "failed"
        job["updated_at"] = _now()
        job["error"] = str(e)
        JOBS[job_id] = job

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/youtube/jobs", response_model=YouTubeJobCreateResp)
def create_youtube_job(
    req: YouTubeJobCreateReq,
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)

    job_id = str(uuid.uuid4())
    now = _now()
    JOBS[job_id] = {
        "job_id": job_id,
        "youtube_url": str(req.youtube_url),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }

    t = threading.Thread(target=_run_youtube_pipeline, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/youtube/jobs/{job_id}", response_model=YouTubeJobResp)
def get_youtube_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job.get("error"),
    }

@app.get("/youtube/jobs/{job_id}/result", response_model=YouTubeJobResultResp)
def get_youtube_job_result(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 结果只在 done/failed 时有意义
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }