import pickle
from typing import Any, Dict
from typing_extensions import Literal
from settings import BASE_DIR
from .model import TextClassifier

FULL_MODEL = TextClassifier(vectorizer_fitted=True)
FULL_MODEL.load_model()


def get_report() -> Dict[str, Any]:
    """Get the report of the model."""
    return FULL_MODEL.get_report()


def infer(sentence: str) -> Literal["positive", "negative"]:
    """Infer the sentiment of a sentence."""
    global FULL_MODEL
    response = FULL_MODEL.inference(sentence)
    print(response)
    return response[0]
