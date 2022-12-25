from fastapi import APIRouter, status
from fastapi.background import BackgroundTasks
from schemas.response_schemas import InferenceResponse
from schemas.request_schemas import InferenceRequest
from typing import Dict, Optional
import asyncio

# from models.external.twinword.inference import infer
from models.scikit_learn.inference import infer


api_router: APIRouter = APIRouter(prefix="/api/v1", tags=["API"])


@api_router.post(
    "/inference",
    response_model=InferenceResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def get_model_inference(request_model: InferenceRequest) -> Dict[str, str]:
    sentiment = infer(request_model.sentence)

    return {
        "sentence": request_model.sentence,
        "sentiment": sentiment,
    }
