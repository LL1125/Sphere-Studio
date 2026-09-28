from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ijnqnwabwqiuddqprdbv.supabase.co").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_vujyfigKbZGNwpjUN6d3YA_nNNRgqf8")
CAPTURE_BUCKET = os.getenv("CAPTURE_BUCKET", "capture-photos")
BA_TIMEOUT = int(os.getenv("BA_TIMEOUT", "70"))
FALLBACK_TIMEOUT = int(os.getenv("FALLBACK_TIMEOUT", "120"))

app = FastAPI(title="Sphere Studio Stitch V2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ll1125.github.io",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_origin_regex=r"https://.*\.github\.io",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Sphere-Path", "X-Sphere-Pipeline", "X-Sphere-QA"],
)


def auth_headers(token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {token}"}
    if extra:
        h.update(extra)
    return h


async def verify_user(token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{SUPABASE_URL}/auth/v1/user", headers=auth_headers(token))
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Supabase session expired. Please start a new capture.")
    return r.json()


async def get_frames(token: str, session_id: str) -> list[dict[str, Any]]:
    params = {
        "select": "shot_index,row_label,sharpness,width,height,storage_path",
        "session_id": f"eq.{session_id}",
        "order": "shot_index.asc",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{SUPABASE_URL}/rest/v1/capture_frames", params=params, headers=auth_headers(token))
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Could not read capture frames: {r.text[:200]}")
    return r.json()


async def download_one(client: httpx.AsyncClient, token: str, row: dict[str, Any], dst: Path) -> None:
    path = row["storage_path"]
    url = f"{SUPABASE_URL}/storage/v1/object/{CAPTURE_BUCKET}/{quote(path, safe='/')}"
    r = await client.get(url, headers=auth_headers(token), timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed for shot {row.get('shot_index')}: HTTP {r.status_code}")
    dst.write_bytes(r.content)


async def download_frames(token: str, rows: list[dict[str, Any]], folder: Path) -> list[Path]:
    sem = asyncio.Semaphore(6)
    outputs: list[Path] = []

    async with httpx.AsyncClient() as client:
        async def task(row: dict[str, Any]) -> Path:
            async with sem:
                shot = int(row["shot_index"])
                out = folder / f"{shot:02d}.jpg"
                await download_one(client, token, row, out)
                return out

        outputs = await asyncio.gather(*(task(r) for r in rows))
    return sorted(outputs)


def run_worker(folder: Path, output: Path, qa: Path) -> str:
    worker = Path(__file__).with_name("stitch_worker.py")
    common = [sys.executable, str(worker), "--input", str(folder), "--output", str(output), "--qa", str(qa)]
    try:
        subprocess.run(common + ["--bundle-adjust"], check=True, timeout=BA_TIMEOUT)
        return "features+homography+bundle-adjust+spherical+seam+exposure+multiband"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        if output.exists():
            output.unlink()
        subprocess.run(common + ["--no-bundle-adjust"], check=True, timeout=FALLBACK_TIMEOUT)
        return "features+homography+spherical+seam+exposure+multiband"


async def upload_panorama(token: str, user_id: str, session_id: str, data: bytes) -> str:
    path = f"{user_id}/{session_id}/panorama-v2.jpg"
    url = f"{SUPABASE_URL}/storage/v1/object/{CAPTURE_BUCKET}/{quote(path, safe='/')}"
    headers = auth_headers(token, {"Content-Type": "image/jpeg", "x-upsert": "true"})
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, content=data)
        if r.status_code >= 400:
            r = await client.put(url, headers=headers, content=data)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Panorama upload failed: {r.text[:200]}")
    return path


@app.get("/health")
async def health():
    return {"ok": True, "service": "sphere-studio-stitch-v2", "version": "2.0.0"}


@app.post("/api/v2/stitch/{session_id}")
async def stitch_session(session_id: str, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Supabase bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user = await verify_user(token)
    rows = await get_frames(token, session_id)

    # V2 capture protocol is exactly 36 frames: 12 horizon + 12 upper + 12 lower.
    if len(rows) != 36:
        raise HTTPException(status_code=409, detail=f"V2 requires 36 uploaded frames; found {len(rows)}")
    shot_ids = [int(r["shot_index"]) for r in rows]
    if shot_ids != list(range(1, 37)):
        raise HTTPException(status_code=409, detail="Capture frame sequence is incomplete")

    work = Path(tempfile.mkdtemp(prefix="sphere-v2-"))
    try:
        frames_dir = work / "frames"
        frames_dir.mkdir()
        await download_frames(token, rows, frames_dir)
        output = work / "panorama-v2.jpg"
        qa_path = work / "qa.json"

        try:
            pipeline = await asyncio.to_thread(run_worker, frames_dir, output, qa_path)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Stitching timed out; please recapture with less camera translation")
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=422, detail=f"Feature-based stitching failed (code {e.returncode})")

        if not output.exists() or output.stat().st_size < 50_000:
            raise HTTPException(status_code=422, detail="Stitching produced no usable panorama")

        data = output.read_bytes()
        storage_path = await upload_panorama(token, user["id"], session_id, data)
        qa = {}
        if qa_path.exists():
            try:
                qa = json.loads(qa_path.read_text("utf-8"))
            except Exception:
                qa = {}
        qa["pipeline"] = pipeline

        return Response(
            data,
            media_type="image/jpeg",
            headers={
                "X-Sphere-Path": storage_path,
                "X-Sphere-Pipeline": pipeline,
                "X-Sphere-QA": json.dumps(qa, ensure_ascii=True, separators=(",", ":"))[:6000],
                "Cache-Control": "no-store",
            },
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
