import pickle
from typing_extensions import Literal
from settings import BASE_DIR
from .model import TextClassifier

FULL_MODEL = TextClassifier(vectorizer_fitted=True)
FULL_MODEL.load_model()


def infer(sentence: str) -> Literal["positive", "negative"]:
    global FULL_MODEL
    response = FULL_MODEL.inference(sentence)
    print(response)
    return response[0]
