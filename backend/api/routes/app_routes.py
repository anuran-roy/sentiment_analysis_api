from fastapi import APIRouter, status
from fastapi.background import BackgroundTasks
from schemas.response_schemas import InferenceResponse
from schemas.request_schemas import InferenceRequest
from typing import Dict, Optional
import pickle
from models.scikit_learn.inference import infer

# from models.scikit_learn.train import train

# from models.external.twinword.inference import infer


app_router: APIRouter = APIRouter(prefix="/app", tags=["App"])


@app_router.post(
    "/inference", response_model=InferenceResponse, status_code=status.HTTP_202_ACCEPTED
)
async def get_model_inference(request_model: InferenceRequest) -> Dict[str, str]:
    sentiment = infer(request_model.sentence)

    return {
        "sentence": request_model.sentence,
        "sentiment": sentiment,
    }


# @app_router.get("/train")
# async def train_model() -> Dict[str, str]:
#     print("Training started")
#     train(
#         mode="file",
#         source_file_type="csv",
#         source_text="/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/data/airline_sentiment_analysis.csv",
#     )
#     return {"status": "Training completed/"}
