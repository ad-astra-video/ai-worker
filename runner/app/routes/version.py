import os

from fastapi import APIRouter
from app.pipelines.base import Version

router = APIRouter()

@router.get("/version", operation_id="version", response_model=Version)
@router.get("/version/", response_model=Version, include_in_schema=False)
def version() -> Version:
    return Version(
        pipeline=os.environ["PIPELINE"],
        model_id=os.environ["MODEL_ID"],
        version=os.environ["VERSION"],
        video_ingress=os.environ.get("VIDEO_INGRESS", True),
        video_egress=os.environ.get("VIDEO_EGRESS", True),
        data_output=os.environ.get("DATA_OUTPUT", True),
    )
