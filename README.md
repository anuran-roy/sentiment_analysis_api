# Sentiment Analysis API

## Tech Stack

- Containerization: Docker
- Backend: FastAPI
- Machine Learning:
    - Scikit-learn, PyTorch (under development)
    - External APIs on RapidAPI
- Frontend: React+TS (Under work)

## Getting Started

### Installation

1. Install docker on your system
2. Navigate to the root folder in the repository through the Command line
3. Run `docker compose up`
4. Yep, we're done!

### Accessing the APIs

To access the frontend, go to `http://localhost:3000` on your PC (provided the port is free)
To access the backend, go to `http://localhost:8000/docs` for the Swagger Documentation.

## Endpoints:

- `/app`: The APIs accessed by the frontend
    - `/app/inference`: The Inference API used by the frontend

- `/api/v1`: Raw REST APIs for direct consumption
    - `/api/v1/inference`: The Inference API for direct consumption