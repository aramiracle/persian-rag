import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from backend.api.dependencies import get_upload_service_dependency
from backend.services.upload_service import UploadService, get_job_status, UPLOAD_JOBS
from backend.core.config import settings

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/csv", status_code=202)
def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_service: UploadService = Depends(get_upload_service_dependency),
):
    """
    Handles massive CSV uploads by streaming the file to a temporary location
    using standard I/O in a threadpool (no async/await for file write).
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    job_id = str(uuid.uuid4())
    
    # 1. Use configured path (No hardcoding)
    temp_dir = Path(settings.upload.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / f"{job_id}.csv"

    # 2. Stream content to disk using standard shutil
    #    Since this is a 'def' route, FastAPI runs it in a threadpool, 
    #    so blocking I/O here is safe and efficient.
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        file.file.close()

    # 3. Initialize Job
    UPLOAD_JOBS.create_sync(job_id)

    # 4. Pass FILE PATH to background task
    background_tasks.add_task(
        upload_service.process_csv_background, 
        job_id, 
        str(file_path)
    )

    return {"job_id": job_id, "message": "Upload started"}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    status = await get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status