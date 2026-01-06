from __future__ import annotations

import os
import glob
import uuid
import shutil
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# TensorFlow oneDNN 관련 로그를 줄이고 싶다면 0으로 설정 (원하면 삭제해도 됨)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Docker WORKDIR 기준 경로 (모든 파일 작업은 이 경로 기준으로 수행)
WORK_DIR = os.environ.get("WORK_DIR", "/app")

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
    enforce_detection: bool = False  # 기본값 False: 얼굴 검출 실패해도 빈 결과 반환
    detector_backend: str | None = None
    dedupe_identity: bool = True  # 같은 identity면 confidence 가장 높은 1개만 남김


class SaveTempFileResponse(BaseModel):
    saved_path: str


class SaveTempFilesResponse(BaseModel):
    files: list[SaveTempFileResponse]
    total: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_path(path: str) -> str:
    """
    경로를 정규화합니다. Windows와 Unix 경로를 모두 처리합니다.
    모든 경로는 WORK_DIR(/app) 기준으로 처리됩니다.
    files/temp/raw에서 반환된 경로를 그대로 사용할 수 있도록 보장합니다.
    """
    if not path:
        return path
    
    # 절대 경로인 경우 그대로 사용 (이미 /app 기준일 수 있음)
    if os.path.isabs(path):
        normalized = os.path.normpath(path)
        # /app 밖의 경로는 /app 기준으로 변환
        if not normalized.startswith(WORK_DIR):
            # 절대 경로지만 /app 밖이면 상대 경로로 간주하고 /app 기준으로 변환
            normalized = os.path.join(WORK_DIR, normalized.lstrip("/"))
            normalized = os.path.normpath(normalized)
    else:
        # 상대 경로는 WORK_DIR 기준으로 변환
        normalized = os.path.join(WORK_DIR, path)
        normalized = os.path.normpath(normalized)
    
    return normalized


def _ensure_exists(path: str, is_dir: bool = False) -> None:
    """
    경로가 존재하는지 확인합니다. 모든 경로는 WORK_DIR(/app) 기준으로 처리됩니다.
    """
    normalized_path = _normalize_path(path)
    if not os.path.exists(normalized_path):
        raise HTTPException(
            status_code=400,
            detail=f"Path not found: {normalized_path} (original: {path}). "
                   f"All paths must be relative to {WORK_DIR}. "
                   f"If this path was returned from /files/temp, ensure the file was saved successfully."
        )
    if is_dir and not os.path.isdir(normalized_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
    if not is_dir and not os.path.isfile(normalized_path):
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")


def _to_deepface_kwargs(model: BaseModel) -> dict[str, Any]:
    # Only pass non-null keys so DeepFace can use its own defaults.
    return {k: v for k, v in model.model_dump().items() if v is not None}


def _safe_makedirs(dir_path: str) -> None:
    """
    WORK_DIR(/app) 기준으로 폴더를 생성합니다.
    """
    normalized_path = _normalize_path(dir_path)
    os.makedirs(normalized_path, exist_ok=True)


def _delete_file_safely(path: str) -> None:
    """파일을 안전하게 삭제합니다."""
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception:
        # 파일 삭제 실패는 무시
        pass


def _delete_directory_safely(dir_path: str) -> None:
    """디렉토리 내의 모든 파일을 안전하게 삭제합니다."""
    try:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass
                for dir_name in dirs:
                    try:
                        shutil.rmtree(os.path.join(root, dir_name))
                    except Exception:
                        pass
    except Exception:
        # 디렉토리 삭제 실패는 무시
        pass


@app.post("/files/temp", response_model=SaveTempFilesResponse)
async def save_temp_file(
    dir_path: str = Form(...),
    files: list[UploadFile] = File(...),
) -> SaveTempFilesResponse:
    """
    dir_path를 받으면 폴더를 만들고, 업로드 파일들을 해당 폴더에 저장합니다.
    여러 파일을 한 번에 업로드할 수 있습니다.
    모든 경로는 /app 기준으로 처리됩니다.
    파일 정리는 사용자가 직접 관리합니다.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    # dir_path를 /app 기준으로 정규화
    normalized_dir_path = _normalize_path(dir_path)
    _safe_makedirs(normalized_dir_path)

    saved_files: list[SaveTempFileResponse] = []
    errors: list[str] = []

    for file in files:
        original = file.filename or "upload"
        _, ext = os.path.splitext(original)
        name = f"{original}"
        saved_path = os.path.join(normalized_dir_path, name)

        # 저장
        try:
            content = await file.read()
            with open(saved_path, "wb") as f:
                f.write(content)
            
            # 반환되는 경로를 정규화하여 find, verify, analyze API에서 바로 사용할 수 있도록 보장
            normalized_saved_path = _normalize_path(saved_path)
            saved_files.append(SaveTempFileResponse(
                saved_path=normalized_saved_path
            ))
        except Exception as e:
            errors.append(f"Failed to save {file.filename}: {str(e)}")
            _delete_file_safely(saved_path)

    if not saved_files and errors:
        raise HTTPException(status_code=500, detail=f"All files failed to save: {'; '.join(errors)}")

    return SaveTempFilesResponse(files=saved_files, total=len(saved_files))







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
    
    # files/temp/raw에서 반환된 경로를 정규화하여 사용
    img1_path = _normalize_path(req.img1_path)
    img2_path = _normalize_path(req.img2_path)
    
    _ensure_exists(img1_path)
    _ensure_exists(img2_path)

    kwargs = _to_deepface_kwargs(req)
    # 정규화된 경로로 교체
    kwargs['img1_path'] = img1_path
    kwargs['img2_path'] = img2_path
    try:
        from deepface import DeepFace

        result = await run_in_threadpool(DeepFace.verify, **kwargs)
        
        # 결과 반환 전에 요청 파일과 비교 파일 삭제
        _delete_file_safely(img1_path)
        _delete_file_safely(img2_path)
        
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
    
    # files/temp/raw에서 반환된 경로를 정규화하여 사용
    img_path = _normalize_path(req.img_path)
    db_path = _normalize_path(req.db_path)
    
    _ensure_exists(img_path, is_dir=False)
    _ensure_exists(db_path, is_dir=True)

    # Check if db_path contains image files (정규화된 경로 사용)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(db_path, ext)))
        image_files.extend(glob.glob(os.path.join(db_path, ext.upper())))
    
    if not image_files:
        raise HTTPException(
            status_code=400,
            detail=f"No image files found in db_path: {db_path} (original: {req.db_path}). "
                   f"Please ensure the directory contains image files (jpg, jpeg, png, bmp, gif). "
                   f"All paths are relative to {WORK_DIR}."
        )

    kwargs = _to_deepface_kwargs(req)
    dedupe_identity = bool(kwargs.pop("dedupe_identity", True))
    # 정규화된 경로로 교체
    kwargs['img_path'] = img_path
    kwargs['db_path'] = db_path

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
        
        # 결과 반환 전에 요청 파일과 db_path 내의 모든 파일 삭제
        _delete_file_safely(img_path)
        _delete_directory_safely(db_path)
        
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
    
    # files/temp/raw에서 반환된 경로를 정규화하여 사용
    img_path = _normalize_path(req.img_path)
    _ensure_exists(img_path)
    
    kwargs = _to_deepface_kwargs(req)
    # 정규화된 경로로 교체
    kwargs['img_path'] = img_path
    try:
        from deepface import DeepFace

        result = await run_in_threadpool(DeepFace.analyze, **kwargs)
        return JSONResponse(content=_json_safe(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e









