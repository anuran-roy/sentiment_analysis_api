import pickle
from typing_extensions import Literal
from settings import BASE_DIR
from .model import TextClassifier


def infer(sentence: str) -> Literal["positive", "negative"]:
    # model = TextClassifier(vectorizer_fit=True)
    # model.load_model(
    #     classifier="/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/backend/api/models/scikit_learn/weights/classifier/best.pkl",
    #     vectorizer="/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/backend/api/models/scikit_learn/weights/vectorizer/best.pkl",
    # )
    # return model.inference(model.vectorize(sentence))

    vectorizer = pickle.load(
        open(
            "/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/backend/api/models/scikit_learn/weights/vectorizer/best.pkl",
            "rb",
        )
    )
    classifier = pickle.load(
        open(
            "/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/backend/api/models/scikit_learn/weights/classifier/best.pkl",
            "rb",
        )
    )

    sentence = vectorizer.transform([sentence])
    response = classifier.predict(sentence)
    print(response)
    return response[0]
