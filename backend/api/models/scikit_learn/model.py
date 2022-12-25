from typing import Any, List, Dict, Tuple, Union, Optional, Set, Literal
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import pickle
import numpy as np
from settings import BASE_DIR
from utils.preprocessing import preprocess_sentence


class TextClassifier:
    def __init__(
        self,
        corpus: Union[np.array, pd.Series, List[str], Tuple[str], Set[str]] = [],
        *args: Optional[List[Any]],
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.corpus: Union[
            np.array, pd.Series, List[str], Tuple[str], Set[str]
        ] = corpus
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            # max_features=2500,
            # min_df=7,
            # max_df=0.8,
            stop_words=stopwords.words("english"),
        )
        self.vectorizer_fitted = kwargs.get("vectorizer_fitted", False)
        self.classifier_model: RandomForestClassifier = RandomForestClassifier(
            random_state=42
        )

    def train_vectorizer(
        self,
    ) -> None:
        self.vectorizer.fit(self.corpus)
        self.vectorizer_fitted = True

    def vectorize(
        self, sentences: Union[pd.Series, np.array, List[str], Tuple[str], Set[str]]
    ) -> np.array:
        print("From vectorize(): ")
        print("Input type = ", type(sentences))
        transform = self.vectorizer.transform(sentences).toarray()
        print("Output type = ", type(transform))
        return transform

    def train(
        self,
        word_vectors: Union[pd.Series, np.ndarray, np.array],
        train_sentiments: pd.Series,
    ):
        self.classifier_model.fit(word_vectors, train_sentiments)

    def test(
        self,
        test_sentences: Union[pd.DataFrame, pd.Series],
        test_sentiments: pd.Series,
    ) -> Dict[str, Any]:
        model_predictions: np.array = self.classifier_model.predict(test_sentences)
        self.model_report: Dict[str, Any] = {
            "accuracy": accuracy_score(test_sentiments, model_predictions),
            "classification_report": classification_report(
                test_sentiments, model_predictions
            ),
            "confusion_matrix": confusion_matrix(test_sentiments, model_predictions),
            # "f1_score": f1_score(test_sentiments, model_predictions),
        }

    def get_report(self) -> Dict[str, Any]:
        return self.model_report

    def inference(
        self,
        sentence: str,
        *args: Optional[List[Any]],
        **kwargs: Optional[Dict[str, Any]]
    ) -> str:
        sentence = preprocess_sentence(sentence)
        if not self.vectorizer_fitted:
            self.train_vectorizer()
        sentence = self.vectorizer.transform(sentence)
        return self.classifier_model.predict(sentence)

    def load_model(self, classifier: str, vectorizer: str) -> None:
        with open(classifier, "rb") as f:
            self.classifier_model = pickle.load(f)

        with open(vectorizer, "rb") as f:
            self.vectorizer = pickle.load(f)

    def save_model(
        self,
        weights_name: str = "best.pkl",
        loc: str = (BASE_DIR / "models" / "scikit_learn" / "weights"),
    ) -> None:
        with open((loc / "classifier" / weights_name), "wb") as f:
            pickle.dump(self.classifier_model, f)
        with open((loc / "vectorizer" / weights_name), "wb") as f:
            pickle.dump(self.vectorizer, f)
