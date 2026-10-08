"""
Sperm Analysis GPU Microservice
Runs as a standalone FastAPI service on the GPU VM.
The main Garbha AI app (CPU VM) proxies requests to this service.
"""

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
    Depends,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import os
import uuid
import asyncio
import threading
import time
import logging
import json
import configparser
from typing import Dict, Optional

import sys

# Ensure the sperm_pipeline package is importable from the nested folder structure
# Structure: sperm_analysis/sperm_pipeline/sperm_pipeline/__init__.py
_pipeline_parent = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sperm_pipeline"
)
if _pipeline_parent not in sys.path:
    sys.path.insert(0, _pipeline_parent)

from sperm_pipeline import SpermAnalysisPipeline, LiveStreamProcessor
from sperm_pipeline.grid_tracking import coordinate_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────
# Model paths: env vars > config.ini > defaults relative to this file
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_config = configparser.ConfigParser()
_config.read(os.path.join(_BASE_DIR, "config.ini"))


def _cfg(section: str, key: str, fallback: str = "") -> str:
    """Read config: env var > config.ini > fallback."""
    env_key = f"SPERM_{section.upper()}_{key.upper()}"
    return os.environ.get(env_key, _config.get(section, key, fallback=fallback))


DETECTION_MODEL_PATH = _cfg(
    "models", "detection_path", os.path.join(_BASE_DIR, "detection_best.pt")
)
MASKRCNN_MODEL_PATH = _cfg(
    "models", "maskrcnn_path", os.path.join(_BASE_DIR, "best_resnet50_transfer_from_101.pth")
)
HNK_MODEL_PATH = _cfg("models", "hnk_path", os.path.join(_BASE_DIR, "HNK_best.pt"))

# Device: "cuda" on GPU VM, "cpu" on CPU VM
DEVICE = os.environ.get(
    "SPERM_DEVICE", _config.get("models", "device", fallback="cuda")
)

# Internal API key shared between CPU VM and GPU VM
API_KEY = os.environ.get("SPERM_API_KEY", _config.get("app", "api_key", fallback=""))
if not API_KEY:
    logger.warning(
        "SPERM_API_KEY not set. Set it via env var or config.ini [app] api_key for production."
    )

UPLOAD_DIR = os.path.join(_BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pipeline params
FRAME_SKIP = int(os.environ.get("SPERM_FRAME_SKIP", _cfg("pipeline", "frame_skip", "1")))
MORPH_EVERY_N = int(os.environ.get("SPERM_MORPH_EVERY_N", _cfg("pipeline", "morph_every_n", "5")))
MASKRCNN_DOWNSCALE = float(os.environ.get("SPERM_MASKRCNN_DOWNSCALE", _cfg("pipeline", "maskrcnn_downscale", "1.0")))
MASKRCNN_THRESHOLD = float(os.environ.get("SPERM_MASKRCNN_THRESHOLD", _cfg("pipeline", "maskrcnn_threshold", "0.7")))
MASKRCNN_SPERM_CLASS_IDS = os.environ.get("SPERM_SPERM_CLASS_IDS", _cfg("models", "sperm_class_ids", "2"))
SLICED_INFERENCE = os.environ.get("SPERM_SLICED_INFERENCE", _cfg("pipeline", "sliced_inference", "false")).lower() not in {"0", "false", "no", "off"}
SLICE_SIZE = int(os.environ.get("SPERM_SLICE_SIZE", _cfg("pipeline", "slice_size", "320")))
SLICE_OVERLAP = float(os.environ.get("SPERM_SLICE_OVERLAP", _cfg("pipeline", "slice_overlap", "0.25")))
TRACKING_METHOD = _cfg("pipeline", "tracking_method", "distance").strip().lower()
DETECTOR_INTERVAL = int(_cfg("pipeline", "detector_interval", "10"))
_um_per_px = _cfg("pipeline", "micrometers_per_pixel", "").strip()
MICROMETERS_PER_PIXEL = float(_um_per_px) if _um_per_px else None

SHOW_TRACKING_GRID = _cfg("pipeline", "show_tracking_grid", "false").lower() in {"1", "true", "yes", "on"}
GRID_ROWS = int(_cfg("pipeline", "grid_rows", "8"))
GRID_COLUMNS = int(_cfg("pipeline", "grid_columns", "8"))
GRID_HISTORY_LENGTH = int(_cfg("pipeline", "grid_history_length", "40"))

# Live camera params
LIVE_CAMERA_INDEX = int(_cfg("live", "camera_index", "0"))
LIVE_RESOLUTION_W = int(_cfg("live", "resolution_w", "1280"))
LIVE_RESOLUTION_H = int(_cfg("live", "resolution_h", "720"))
LIVE_TARGET_FPS = int(_cfg("live", "target_fps", "30"))
LIVE_JPEG_QUALITY = int(_cfg("live", "jpeg_quality", "80"))
LIVE_PUBLIC_IP = _cfg("live", "public_ip", "")
LIVE_PUBLIC_PORT = int(_cfg("live", "public_port", "5002"))

# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Sperm Analysis - GPU Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (mobile camera page, etc.)
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── Auth ────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate internal API key. Skipped if API_KEY is not configured."""
    if not API_KEY:
        return  # no key configured = allow (dev mode)
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── State ───────────────────────────────────────────────────────────────
progress_tracker: Dict[str, Dict] = {}
processing_jobs: Dict[str, Dict] = {}
pipeline: Optional[SpermAnalysisPipeline] = None
live_processor: Optional[LiveStreamProcessor] = None
# WebSocket sessions: session_id -> LiveStreamProcessor
ws_sessions: Dict[str, LiveStreamProcessor] = {}
ws_connections: Dict[str, WebSocket] = {}


def initialize_pipeline():
    global pipeline
    try:
        pipeline = SpermAnalysisPipeline(
            detection_model_path=DETECTION_MODEL_PATH,
            maskrcnn_model_path=MASKRCNN_MODEL_PATH,
            hnk_model_path=HNK_MODEL_PATH,
            device=DEVICE,
            frame_skip=FRAME_SKIP,
            morph_every_n=MORPH_EVERY_N,
            maskrcnn_downscale=MASKRCNN_DOWNSCALE,
            maskrcnn_threshold=MASKRCNN_THRESHOLD,
            maskrcnn_sperm_class_ids=MASKRCNN_SPERM_CLASS_IDS,
            sliced_inference=SLICED_INFERENCE,
            slice_size=SLICE_SIZE,
            slice_overlap=SLICE_OVERLAP,
            tracking_method=TRACKING_METHOD,
            detector_interval=DETECTOR_INTERVAL,
            micrometers_per_pixel=MICROMETERS_PER_PIXEL,
            show_tracking_grid=SHOW_TRACKING_GRID, grid_rows=GRID_ROWS,
            grid_columns=GRID_COLUMNS, grid_history_length=GRID_HISTORY_LENGTH,
        )
        logger.info(
            f"Pipeline initialized successfully (device={DEVICE}, "
            f"frame_skip={FRAME_SKIP}, maskrcnn_threshold={MASKRCNN_THRESHOLD}, "
            f"sperm_class_ids={MASKRCNN_SPERM_CLASS_IDS}, "
            f"sliced_inference={SLICED_INFERENCE}, slice_size={SLICE_SIZE}, "
            f"slice_overlap={SLICE_OVERLAP}, tracking_method={TRACKING_METHOD}, "
            f"detector_interval={DETECTOR_INTERVAL})"
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None


def _set_progress(job_id: str, message: str, progress: int, status: str):
    progress_tracker[job_id] = {
        "status": status,
        "message": message,
        "progress": progress,
        "timestamp": time.time(),
    }


def process_video_async(video_path: str, job_dir: str, job_id: str):
    def progress_callback(message: str, percent: int):
        _set_progress(job_id, message, percent, "processing")

    try:
        if pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        _set_progress(job_id, "Starting analysis...", 0, "processing")
        results = pipeline.process_video(video_path, job_dir, progress_callback)
        processed_video_path = results["output_paths"]["processed_video"]
        inference_video_path = results["output_paths"]["inference_video"]
        processing_jobs[job_id] = {
            "status": "completed",
            "video_path": inference_video_path,
            "processed_video_path": processed_video_path,
            "inference_video_path": inference_video_path,
            "json_path": results["output_paths"]["json_path"],
            "csv_path": results["output_paths"]["csv_path"],
            "trajectories_json_path": results["output_paths"]["trajectories_path"],
            "inference_tracking_csv_path": results["output_paths"].get("inference_tracking_path"),
            "analysis_id_diagnostics_path": results["output_paths"].get("analysis_id_diagnostics_path"),
            "tracking_quality_summary_path": results["output_paths"].get("tracking_quality_summary_path"),
            "identity_registry_path": results["output_paths"].get("identity_registry_path"),
            "identity_quality_events_path": results["output_paths"].get("identity_quality_events_path"),
            **{key: results["output_paths"].get(key) for key in (
                "grid_tracking_csv_path", "grid_tracking_json_path", "grid_tracking_video_path")},
            "results": results["analysis_results"],
            "video_properties": results["video_properties"],
            "timestamp": time.time(),
        }
        _set_progress(job_id, "Analysis completed successfully!", 100, "completed")
    except Exception as e:
        logger.exception("Processing failed")
        _set_progress(job_id, f"Error: {str(e)}", 0, "error")
        processing_jobs[job_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }


# ── Lifecycle ───────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    initialize_pipeline()

    def periodic_cleanup():
        while True:
            time.sleep(1800)
            current_time = time.time()
            max_age = 3600
            to_remove = [
                jid
                for jid, job in processing_jobs.items()
                if current_time - job.get("timestamp", 0) > max_age
            ]
            for jid in to_remove:
                processing_jobs.pop(jid, None)
                progress_tracker.pop(jid, None)

    t = threading.Thread(target=periodic_cleanup, daemon=True)
    t.start()


# ── Routes ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint for the CPU VM to verify GPU service is up."""
    return {"status": "ok", "pipeline_ready": pipeline is not None, "device": DEVICE}


@app.post("/process", dependencies=[Depends(verify_api_key)])
async def process(video: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    filename = video.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid file")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"mp4", "avi", "mov", "mkv"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use mp4, avi, mov, or mkv.",
        )

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    in_path = os.path.join(job_dir, filename)

    with open(in_path, "wb") as f:
        f.write(await video.read())

    _set_progress(job_id, "Video uploaded, starting processing...", 0, "queued")

    t = threading.Thread(
        target=process_video_async, args=(in_path, job_dir, job_id), daemon=True
    )
    t.start()

    return JSONResponse(
        {"job_id": job_id, "status": "processing", "message": "Processing started"}
    )


@app.get("/progress/{job_id}", dependencies=[Depends(verify_api_key)])
def get_progress(job_id: str):
    if job_id not in progress_tracker:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(progress_tracker[job_id])


@app.get("/progress_stream/{job_id}", dependencies=[Depends(verify_api_key)])
def progress_stream(job_id: str):
    async def event_generator():
        while True:
            if job_id in progress_tracker:
                yield f"data: {json.dumps(progress_tracker[job_id])}\n\n"
                if progress_tracker[job_id]["status"] in ["completed", "error"]:
                    break
            else:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/results/{job_id}", dependencies=[Depends(verify_api_key)])
def get_results(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found or not completed")
    job = processing_jobs[job_id]
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    video_url = f"/outputs/{job_id}/{os.path.basename(job['video_path'])}"
    inference_url = f"/outputs/{job_id}/{os.path.basename(job['inference_video_path'])}"
    processed_path = job.get("processed_video_path") or job.get("video_path")
    processed_url = f"/outputs/{job_id}/{os.path.basename(processed_path)}" if processed_path else None
    json_url = f"/outputs/{job_id}/{os.path.relpath(job['json_path'], os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"
    csv_url = f"/outputs/{job_id}/{os.path.relpath(job['csv_path'], os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    trajectories_url = None
    tjp = job.get("trajectories_json_path")
    if tjp and os.path.exists(tjp):
        trajectories_url = f"/outputs/{job_id}/{os.path.relpath(tjp, os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    inference_tracking_url = None
    itp = job.get("inference_tracking_csv_path")
    if itp and os.path.exists(itp):
        inference_tracking_url = f"/outputs/{job_id}/{os.path.relpath(itp, os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    analysis_id_diagnostics_url = None
    adp = job.get("analysis_id_diagnostics_path")
    if adp and os.path.exists(adp):
        analysis_id_diagnostics_url = f"/outputs/{job_id}/{os.path.relpath(adp, os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    quality_url = None
    quality_path = job.get("tracking_quality_summary_path")
    if quality_path and os.path.exists(quality_path):
        quality_url = f"/outputs/{job_id}/{os.path.relpath(quality_path, os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    registry_url = None
    registry_path = job.get("identity_registry_path")
    if registry_path and os.path.exists(registry_path):
        registry_url = f"/outputs/{job_id}/{os.path.relpath(registry_path, os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    identity_events_url = None
    identity_events_path = job.get("identity_quality_events_path")
    if identity_events_path and os.path.exists(identity_events_path):
        identity_events_url = f"/outputs/{job_id}/{os.path.relpath(identity_events_path, os.path.join(OUTPUT_DIR, job_id)).replace(chr(92), '/')}"

    grid_urls = {}
    for key in ("grid_tracking_csv", "grid_tracking_json", "grid_tracking_video"):
        path = job.get(key + "_path")
        grid_urls[key] = (f"/outputs/{job_id}/{os.path.relpath(path, os.path.join(OUTPUT_DIR, job_id))}"
                          if path and os.path.isfile(path) else None)
    return JSONResponse(
        {
            **grid_urls,
            "job_id": job_id,
            "video": video_url,
            "inference_video": inference_url,
            "processed_video": processed_url,
            "summary_json": json_url,
            "summary_csv": csv_url,
            "trajectories_json": trajectories_url,
            "inference_tracking_csv": inference_tracking_url,
            "analysis_id_diagnostics": analysis_id_diagnostics_url,
            "tracking_quality_summary": quality_url,
            "application_identity_registry": registry_url,
            "application_identity_quality_events": identity_events_url,
            "results": job["results"],
            "video_properties": job.get("video_properties", {}),
        }
    )


@app.get("/diagnostics/{job_id}/sperm/{application_id}", dependencies=[Depends(verify_api_key)])
def get_grid_coordinate_history(job_id: str, application_id: int):
    """Video-local canonical identity lookup; also works for saved completed jobs."""
    if application_id < 1 or not job_id or any(v in job_id for v in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid job or application ID")
    from pathlib import Path
    base = Path(OUTPUT_DIR).resolve()
    path = (base / job_id / "meta" / "grid_tracking_coordinates.json").resolve()
    if not path.is_relative_to(base):
        raise HTTPException(status_code=403, detail="Access denied")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Grid diagnostics not available for this job")
    try:
        return coordinate_history(path, application_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Canonical identity not found")


@app.delete("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def delete_job(job_id: str):
    """Delete a job's output directory and any in-memory state."""
    import shutil

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="Invalid job id")

    job_dir = os.path.join(OUTPUT_DIR, job_id)
    real_base = os.path.realpath(OUTPUT_DIR)
    real_dir = os.path.realpath(job_dir)
    if not real_dir.startswith(real_base):
        raise HTTPException(status_code=403, detail="Access denied")

    removed = False
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
        removed = True

    processing_jobs.pop(job_id, None)
    progress_tracker.pop(job_id, None)

    return JSONResponse({"job_id": job_id, "deleted": removed})


@app.get("/outputs/{job_id}/{path:path}", dependencies=[Depends(verify_api_key)])
def serve_output(job_id: str, path: str, request: Request):
    dir_path = os.path.join(OUTPUT_DIR, job_id)
    file_path = os.path.join(dir_path, path)

    # Prevent path traversal
    real_base = os.path.realpath(dir_path)
    real_file = os.path.realpath(file_path)
    if not real_file.startswith(real_base):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    range_header = request.headers.get("range") or request.headers.get("Range")
    file_size = os.path.getsize(file_path)

    if range_header and file_path.lower().endswith(
        (".mp4", ".m4v", ".mov", ".webm", ".avi")
    ):
        try:
            bytes_unit, range_spec = range_header.split("=", 1)
            if bytes_unit.strip().lower() == "bytes":
                start_str, end_str = (range_spec.split("-", 1) + [""])[:2]
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                start = max(0, start)
                end = min(file_size - 1, end)
                if start > end:
                    start, end = 0, file_size - 1
                length = end - start + 1
                with open(file_path, "rb") as f:
                    f.seek(start)
                    data = f.read(length)
                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                }
                return Response(
                    content=data,
                    status_code=206,
                    media_type=_guess_mime(file_path),
                    headers=headers,
                )
        except Exception:
            pass

    resp = FileResponse(file_path, media_type=_guess_mime(file_path))
    resp.headers["Accept-Ranges"] = "bytes"
    return resp


# ── Live Camera Routes ─────────────────────────────────────────────────


@app.get("/live/cameras", dependencies=[Depends(verify_api_key)])
def list_cameras():
    """List available camera devices (USB cameras, mobile phones, etc.)."""
    from sperm_pipeline.live_stream import LiveStreamProcessor
    from sperm_pipeline.detection import SpermDetector

    # Use a temporary processor just for camera listing
    tmp = LiveStreamProcessor.__new__(LiveStreamProcessor)
    cameras = tmp.list_cameras()
    return JSONResponse({"cameras": cameras})


@app.post("/live/start", dependencies=[Depends(verify_api_key)])
def live_start(camera_index: int = LIVE_CAMERA_INDEX):
    """
    Start live camera capture and real-time sperm detection+tracking.

    Query params:
        camera_index: USB camera device index (default from config)
    """
    global live_processor
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if live_processor is not None and live_processor.is_running:
        raise HTTPException(
            status_code=409, detail="Live stream already running. Stop it first."
        )

    live_processor = LiveStreamProcessor(
        detector=pipeline.detector,
        device_index=camera_index,
        resolution=(LIVE_RESOLUTION_W, LIVE_RESOLUTION_H),
        target_fps=LIVE_TARGET_FPS,
    )
    try:
        live_processor.start()
    except RuntimeError as e:
        live_processor = None
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(
        {
            "status": "started",
            "camera_index": camera_index,
            "resolution": f"{LIVE_RESOLUTION_W}x{LIVE_RESOLUTION_H}",
            "target_fps": LIVE_TARGET_FPS,
            "stream_url": "/live/stream",
            "stats_url": "/live/stats",
        }
    )


@app.post("/live/stop", dependencies=[Depends(verify_api_key)])
def live_stop():
    """Stop the live camera stream and return final motility summary."""
    global live_processor
    if live_processor is None:
        raise HTTPException(status_code=404, detail="No live stream running")

    summary = live_processor.get_motility_summary()
    live_processor.close_camera()
    live_processor = None
    return JSONResponse({"status": "stopped", "summary": summary})


@app.get("/live/stream", dependencies=[Depends(verify_api_key)])
def live_stream():
    """
    MJPEG stream of the live annotated camera feed.
    Open this URL in a browser <img> tag or video player.
    """
    if live_processor is None or not live_processor.is_running:
        raise HTTPException(status_code=404, detail="No live stream running")

    def mjpeg_generator():
        while live_processor is not None and live_processor.is_running:
            jpeg = live_processor.get_jpeg_frame(quality=LIVE_JPEG_QUALITY)
            if jpeg is not None:
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.01)

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/live/frame", dependencies=[Depends(verify_api_key)])
def live_frame():
    """Get a single JPEG snapshot of the current annotated frame."""
    if live_processor is None or not live_processor.is_running:
        raise HTTPException(status_code=404, detail="No live stream running")
    jpeg = live_processor.get_jpeg_frame(quality=LIVE_JPEG_QUALITY)
    if jpeg is None:
        raise HTTPException(status_code=204, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/live/stats", dependencies=[Depends(verify_api_key)])
def live_stats():
    """Get current tracking statistics from the live stream."""
    if live_processor is None:
        raise HTTPException(status_code=404, detail="No live stream running")
    return JSONResponse(live_processor.get_stats())


@app.get("/live/motility", dependencies=[Depends(verify_api_key)])
def live_motility():
    """Get motility analysis summary for all tracked sperm in the live session."""
    if live_processor is None:
        raise HTTPException(status_code=404, detail="No live stream running")
    return JSONResponse(live_processor.get_motility_summary())


@app.post("/live/reset", dependencies=[Depends(verify_api_key)])
def live_reset():
    """Reset tracking state (start fresh while keeping camera open)."""
    if live_processor is None:
        raise HTTPException(status_code=404, detail="No live stream running")
    live_processor.reset_tracking()
    return JSONResponse({"status": "tracking_reset"})


# ── QR Code + WebSocket Camera Routes ──────────────────────────────────


def _get_public_base() -> tuple:
    """Get the public IP and port from config, with fallback."""
    ip = LIVE_PUBLIC_IP
    if not ip:
        import socket

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "127.0.0.1"
    return ip, LIVE_PUBLIC_PORT


@app.get("/live/qr", dependencies=[Depends(verify_api_key)])
def generate_qr(session: Optional[str] = None):
    """
    Generate a QR code that the mobile phone can scan to open the camera page.
    Uses public_ip and public_port from config.ini [live] section.

    Query params:
        session: Optional session ID. If provided, reuses it (must match /qr/info).
                 If omitted, generates a new one.

    Returns:
        QR code as a PNG image.
    """
    import io

    try:
        import qrcode
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="qrcode library not installed. Run: pip install qrcode[pil]",
        )

    server_ip, port = _get_public_base()
    session_id = session or str(uuid.uuid4())[:8]
    # Use HTTPS - required for getUserMedia on mobile browsers
    camera_url = (
        f"https://{server_ip}:{port}/static/mobile_camera.html?session={session_id}"
    )

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(camera_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    logger.info(f"QR generated: {camera_url} (session={session_id})")

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Camera-URL": camera_url,
            "X-Session-ID": session_id,
        },
    )


@app.get("/live/qr/info", dependencies=[Depends(verify_api_key)])
def qr_info():
    """
    Return the camera URL and session info as JSON (for programmatic use).
    Uses public_ip and public_port from config.ini [live] section.
    """
    server_ip, port = _get_public_base()
    session_id = str(uuid.uuid4())[:8]
    camera_url = (
        f"https://{server_ip}:{port}/static/mobile_camera.html?session={session_id}"
    )

    return JSONResponse(
        {
            "camera_url": camera_url,
            "session_id": session_id,
            "server_ip": server_ip,
            "port": port,
            "ws_url": f"wss://{server_ip}:{port}/live/ws/{session_id}",
            "stream_url": f"https://{server_ip}:{port}/live/ws/{session_id}/stream",
        }
    )


@app.websocket("/live/ws/{session_id}")
async def websocket_camera(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for receiving camera frames from the mobile browser.
    The phone sends JPEG frames as binary messages, the server processes them
    and sends back stats as JSON text messages.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: session={session_id}")

    if pipeline is None:
        await websocket.send_json(
            {"type": "error", "message": "Pipeline not initialized"}
        )
        await websocket.close()
        return

    # Create a processor for this session
    processor = LiveStreamProcessor(
        detector=pipeline.detector,
        device_index=0,
        resolution=(LIVE_RESOLUTION_W, LIVE_RESOLUTION_H),
        target_fps=LIVE_TARGET_FPS,
    )
    processor.start_websocket_mode()
    ws_sessions[session_id] = processor
    ws_connections[session_id] = websocket

    await websocket.send_json({"type": "connected", "session_id": session_id})

    import concurrent.futures

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive()

            # Binary message = JPEG frame from mobile camera
            if "bytes" in data and data["bytes"]:
                jpeg_bytes = data["bytes"]
                # Run detection in a thread so it doesn't block the async event loop
                # This allows HTTP stats requests to be served while processing
                await loop.run_in_executor(
                    _executor, processor.process_frame, jpeg_bytes
                )

                # Send stats back to the mobile app
                stats = processor.get_stats()
                if stats:
                    await websocket.send_json(stats)

            # Text message = config or control
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "config":
                        logger.info(f"Session {session_id} config: {msg}")
                    elif msg.get("type") == "reset":
                        processor.reset_tracking()
                        await websocket.send_json({"type": "reset_ok"})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error (session={session_id}): {e}")
    finally:
        processor.stop_websocket_mode()
        ws_sessions.pop(session_id, None)
        ws_connections.pop(session_id, None)


@app.post("/live/ws/{session_id}/disconnect")
async def ws_session_disconnect(session_id: str):
    """Close a WebSocket session. This disconnects the phone."""
    ws = ws_connections.get(session_id)
    if ws:
        try:
            await ws.send_json({"type": "disconnect", "reason": "Session ended by user"})
            await ws.close()
        except Exception:
            pass
    ws_sessions.pop(session_id, None)
    ws_connections.pop(session_id, None)
    logger.info(f"Session {session_id} disconnected by user")
    return JSONResponse({"status": "disconnected", "session_id": session_id})


@app.get("/live/ws/{session_id}/stream")
def ws_session_stream(session_id: str):
    """
    MJPEG stream of the annotated feed for a WebSocket camera session.
    View this on a PC/laptop browser while the phone streams.
    """
    if session_id not in ws_sessions:
        raise HTTPException(
            status_code=404, detail="Session not found. Is the phone connected?"
        )

    processor = ws_sessions[session_id]

    def mjpeg_generator():
        while session_id in ws_sessions:
            jpeg = processor.get_jpeg_frame(quality=LIVE_JPEG_QUALITY)
            if jpeg is not None:
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.03)

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/live/ws/{session_id}/stats")
def ws_session_stats(session_id: str):
    """Get tracking stats for a WebSocket camera session."""
    if session_id not in ws_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(ws_sessions[session_id].get_stats())


@app.get("/live/ws/{session_id}/motility")
def ws_session_motility(session_id: str):
    """Get motility summary for a WebSocket camera session."""
    if session_id not in ws_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(ws_sessions[session_id].get_motility_summary())


@app.get("/live/ws/sessions")
def ws_list_sessions():
    """List all active WebSocket camera sessions."""
    sessions = []
    for sid, proc in ws_sessions.items():
        stats = proc.get_stats()
        sessions.append(
            {
                "session_id": sid,
                "is_running": proc.is_running,
                "total_tracked": stats.get("total_tracked", 0),
                "frame_idx": stats.get("frame_idx", 0),
            }
        )
    return JSONResponse({"sessions": sessions})


def _guess_mime(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("mp4", "m4v"):
        return "video/mp4"
    if ext == "mov":
        return "video/quicktime"
    if ext == "webm":
        return "video/webm"
    if ext == "avi":
        return "video/x-msvideo"
    if ext == "json":
        return "application/json"
    if ext == "csv":
        return "text/csv"
    return "application/octet-stream"


# ── Run server ─────────────────────────────────────────────────────────
# Run with: python fastapi_app.py
# Single HTTPS server on port 5002. Both CPU VM and phone connect here.
# CPU backend uses verify=False for the self-signed cert.

if __name__ == "__main__":
    import uvicorn

    ssl_cert = os.path.join(_BASE_DIR, "ssl", "cert.pem")
    ssl_key = os.path.join(_BASE_DIR, "ssl", "key.pem")
    has_ssl = os.path.exists(ssl_cert) and os.path.exists(ssl_key)

    if has_ssl:
        logger.info("Starting HTTPS server on port 5002 (SSL enabled)")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=5002,
            ssl_keyfile=ssl_key,
            ssl_certfile=ssl_cert,
            log_level="info",
        )
    else:
        logger.warning(
            "SSL certs not found. Mobile camera won't work. Run: python generate_ssl.py"
        )
        logger.info("Starting HTTP server on port 5002 (no SSL)")
        uvicorn.run(app, host="0.0.0.0", port=5002, log_level="info")
