FROM python:3.10.9-bullseye
# RUN pip install poetry
WORKDIR /usr/src/backend
RUN pip install uvicorn poetry
COPY ./pyproject.toml /backend
COPY ./requirements.txt /backend
# RUN poetry install --no-root
RUN pip install -r requirements.txt
WORKDIR /backend/api
COPY ./api /backend/api
RUN cd /backend/api
RUN python3 -c "import nltk;nltk.download('stopwords');nltk.download('punkt');nltk.download('wordnet');)"
# RUN python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]