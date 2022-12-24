import sys
from settings import BASE_DIR
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(BASE_DIR.absolute())

from fastapi import FastAPI

from routes.rest_api_routes import api_router
from routes.app_routes import app_router

app = FastAPI(
    title="Sentiment Analysis Model API",
    description="This is the Sentiment Analysis API for the completion of the ML Internship Task given by TrueFoundry.",
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=app_router)
app.include_router(router=api_router)
