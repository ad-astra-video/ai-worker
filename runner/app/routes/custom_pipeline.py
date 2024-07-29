import logging
import os
import random, json

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, ImageResponse, http_error, image_to_data_url
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router = APIRouter()

logger = logging.getLogger(__name__)


class CustomPipelineParams(BaseModel):
    model_id: str = ""
    args: str


RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post("/custom-pipeline", responses=RESPONSES)
@router.post(
    "/custom-pipeline/",
    responses=RESPONSES,
    include_in_schema=False,
)
async def custom_pipeline(
    params: CustomPipelineParams,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{params.model_id}"
            ),
        )

    args = json.loads(params.args)

    output = pipeline(**args)

    return {"output": output}
