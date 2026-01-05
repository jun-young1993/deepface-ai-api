from __future__ import annotations

import os
import glob
import uuid
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# TensorFlow oneDNN 관련 로그를 줄이고 싶다면 0으로 설정 (원하면 삭제해도 됨)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

app = FastAPI(title="DeepFace API", version="0.1.0")

# CORS 설정 (필요시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerifyRequest(BaseModel):
    # Either provide server-accessible paths OR upload files via multipart endpoints.
    img1_path: str | None = None
    img2_path: str | None = None
    model_name: str | None = None
    detector_backend: str | None = None
    distance_metric: str | None = None
    enforce_detection: bool = True


class AnalyzeRequest(BaseModel):
    img_path: str | None = None
    actions: list[Literal["age", "gender", "race", "emotion"]] = Field(
        default_factory=lambda: ["age", "gender", "race", "emotion"]
    )
    detector_backend: str | None = None
    enforce_detection: bool = True

class FindRequest(BaseModel):
    img_path: str | None = None
    db_path: str | None = None
    model_name: str | None = None
    distance_metric: str | None = None
    enforce_detection: bool = True
    detector_backend: str | None = None
    dedupe_identity: bool = True  # 같은 identity면 confidence 가장 높은 1개만 남김


class SaveTempFileResponse(BaseModel):
    saved_path: str
    expires_in_seconds: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _ensure_exists(path: str, is_dir: bool = False) -> None:
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")
    if is_dir and not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
    if not is_dir and not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")


def _to_deepface_kwargs(model: BaseModel) -> dict[str, Any]:
    # Only pass non-null keys so DeepFace can use its own defaults.
    return {k: v for k, v in model.model_dump().items() if v is not None}


def _safe_makedirs(dir_path: str) -> None:
    # 필요한 경우 폴더 생성
    os.makedirs(dir_path, exist_ok=True)


def _delete_file_safely(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        # 임시 파일 삭제 실패는 무시
        pass


@app.post("/files/temp", response_model=SaveTempFileResponse)
async def save_temp_file(
    dir_path: str = Form(...),
    file: UploadFile = File(...),
    ttl_seconds: int = Form(600),
) -> SaveTempFileResponse:
    """
    dir_path를 받으면 폴더를 만들고, 업로드 파일을 해당 폴더에 '임시'로 저장합니다.
    ttl_seconds 후에 백그라운드로 삭제를 시도합니다.
    """
    if ttl_seconds < 0 or ttl_seconds > 60 * 60 * 24:
        raise HTTPException(status_code=400, detail="ttl_seconds must be between 0 and 86400.")

    _safe_makedirs(dir_path)

    original = file.filename or "upload"
    _, ext = os.path.splitext(original)
    name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(dir_path, name)

    # 저장
    try:
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}") from e

    # 임시 삭제: ttl_seconds == 0이면 즉시 삭제
    if ttl_seconds == 0:
        _delete_file_safely(saved_path)

    return SaveTempFileResponse(saved_path=saved_path, expires_in_seconds=ttl_seconds)


async def _save_raw_body_to_path(request: Request, dest_path: str) -> int:
    """
    Save request body (raw bytes) to dest_path using streaming.
    Returns total bytes written.
    """
    total = 0
    with open(dest_path, "wb") as f:
        async for chunk in request.stream():
            if not chunk:
                continue
            f.write(chunk)
            total += len(chunk)
    return total


@app.post("/files/temp/raw", response_model=SaveTempFileResponse)
async def save_temp_file_raw(
    request: Request,
    dir_path: str,
    filename: str | None = None,
    ttl_seconds: int = 600,
) -> SaveTempFileResponse:
    """
    raw 바이너리(application/octet-stream)를 바디로 받아서,
    dir_path 폴더를 만들고 임시 파일로 저장합니다.

    호출 예:
      curl -X POST "http://127.0.0.1:8000/files/temp/raw?dir_path=C:/tmp&filename=a.jpg" \
        -H "Content-Type: application/octet-stream" \
        --data-binary "@C:/path/to/local.jpg"
    """
    if ttl_seconds < 0 or ttl_seconds > 60 * 60 * 24:
        raise HTTPException(status_code=400, detail="ttl_seconds must be between 0 and 86400.")

    _safe_makedirs(dir_path)

    ext = ""
    if filename:
        _, ext = os.path.splitext(filename)
    name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(dir_path, name)

    try:
        written = await _save_raw_body_to_path(request, saved_path)
        if written == 0:
            _delete_file_safely(saved_path)
            raise HTTPException(status_code=400, detail="Request body is empty.")
    except HTTPException:
        raise
    except Exception as e:
        _delete_file_safely(saved_path)
        raise HTTPException(status_code=500, detail=f"Failed to save raw body: {e}") from e

    # ttl_seconds == 0이면 즉시 삭제
    if ttl_seconds == 0:
        _delete_file_safely(saved_path)

    return SaveTempFileResponse(saved_path=saved_path, expires_in_seconds=ttl_seconds)

def _json_safe(data: Any) -> Any:
    """
    DeepFace results may contain numpy scalars (e.g., numpy.float32) or pandas DataFrames
    which FastAPI cannot serialize by default. Convert them to JSON-friendly types.
    """
    # Handle pandas DataFrame (from DeepFace.find)
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], pd.DataFrame):
            return [df.to_dict(orient='records') for df in data]
    except ImportError:
        pass
    except Exception:
        pass
    
    custom = {}
    try:
        import numpy as np  # type: ignore

        custom = {
            np.generic: lambda v: v.item(),
            np.ndarray: lambda v: v.tolist(),
        }
    except Exception:
        # numpy not available; fall back to default encoder
        custom = {}

    return jsonable_encoder(data, custom_encoder=custom)


def _dedupe_find_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    DeepFace.find 결과에서 동일 identity(파일 경로)가 여러 번 나올 수 있어
    confidence가 가장 높은 1개만 남깁니다. (동률이면 distance가 낮은 걸 우선)
    """
    best: dict[str, dict[str, Any]] = {}

    def key(item: dict[str, Any]) -> tuple[float, float]:
        conf = float(item.get("confidence", -1))
        dist = float(item.get("distance", float("inf")))
        return (conf, -dist)

    for item in records:
        identity = str(item.get("identity", "")).strip()
        if not identity:
            continue
        if identity not in best or key(item) > key(best[identity]):
            best[identity] = item

    deduped = list(best.values())
    deduped.sort(
        key=lambda x: (-float(x.get("confidence", -1)), float(x.get("distance", float("inf"))))
    )
    return deduped


def _dedupe_find_payload(payload: Any) -> Any:
    # payload is usually List[List[Dict]] (source image 내 여러 얼굴이면 바깥 리스트 길이 > 1)
    if isinstance(payload, list) and payload and all(isinstance(x, list) for x in payload):
        out: list[Any] = []
        for face_matches in payload:
            if isinstance(face_matches, list) and all(isinstance(i, dict) for i in face_matches):
                out.append(_dedupe_find_records(face_matches))
            else:
                out.append(face_matches)
        return out

    # payload could also be List[Dict]
    if isinstance(payload, list) and (not payload or all(isinstance(i, dict) for i in payload)):
        return _dedupe_find_records(payload) if payload else payload

    return payload


@app.post("/verify")
async def verify(req: VerifyRequest) -> Any:
    if not req.img1_path or not req.img2_path:
        raise HTTPException(
            status_code=400,
            detail="img1_path and img2_path are required.",
        )
    _ensure_exists(req.img1_path)
    _ensure_exists(req.img2_path)

    kwargs = _to_deepface_kwargs(req)
    try:
        from deepface import DeepFace

        result = await run_in_threadpool(DeepFace.verify, **kwargs)
        return JSONResponse(content=_json_safe(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/find")
async def find(req: FindRequest) -> Any:
    if not req.img_path or not req.db_path:
        raise HTTPException(
            status_code=400,
            detail="img_path and db_path are required.",
        )
    _ensure_exists(req.img_path, is_dir=False)
    _ensure_exists(req.db_path, is_dir=True)

    # Check if db_path contains image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(req.db_path, ext)))
        image_files.extend(glob.glob(os.path.join(req.db_path, ext.upper())))
    
    if not image_files:
        raise HTTPException(
            status_code=400,
            detail=f"No image files found in db_path: {req.db_path}. "
                   f"Please ensure the directory contains image files (jpg, jpeg, png, bmp, gif)."
        )

    kwargs = _to_deepface_kwargs(req)
    dedupe_identity = bool(kwargs.pop("dedupe_identity", True))

    try:
        from deepface import DeepFace

        result = await run_in_threadpool(DeepFace.find, **kwargs)
        # Handle empty results gracefully
        if not result or (isinstance(result, list) and len(result) == 0):
            return JSONResponse(content=[])
        
        # Check if result contains empty DataFrames
        try:
            import pandas as pd
            if isinstance(result, list):
                filtered_result = []
                for item in result:
                    if isinstance(item, pd.DataFrame):
                        if len(item) > 0:
                            filtered_result.append(item)
                    else:
                        filtered_result.append(item)
                if not filtered_result:
                    return JSONResponse(content=[])
                result = filtered_result
        except ImportError:
            pass
        
        payload = _json_safe(result)
        if dedupe_identity:
            payload = _dedupe_find_payload(payload)
        return JSONResponse(content=payload)
    except ValueError as e:
        # Handle pandas DataFrame creation errors
        error_msg = str(e)
        if "Length of values" in error_msg and "does not match length of index" in error_msg:
            # This usually means no faces detected - return empty result instead of error
            return JSONResponse(content=[])
        raise HTTPException(status_code=400, detail=f"Validation error: {error_msg}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in DeepFace.find: {str(e)}") from e


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Any:
    if not req.img_path:
        raise HTTPException(
            status_code=400,
            detail="img_path is required.",
        )
    _ensure_exists(req.img_path)

    kwargs = _to_deepface_kwargs(req)
    try:
        from deepface import DeepFace

        result = await run_in_threadpool(DeepFace.analyze, **kwargs)
        return JSONResponse(content=_json_safe(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e









