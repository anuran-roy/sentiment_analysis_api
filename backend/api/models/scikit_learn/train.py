# from typing import Union, Literal, List, Tuple, Dict, Any, Optional
# from utils.preprocessing import preprocess_sentence

# from settings import BASE_DIR
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# from models.scikit_learn.model import TextClassifier


# def train(
#     source_text: Union[pd.Series, pd.DataFrame, np.array, str],
#     mode: Literal["pandas", "numpy", "file"] = "pandas",
#     source_file_type: Literal["csv", "parquet", "json", "sql", "xml"] = "csv",
#     *args: Optional[List[Any]],
#     **kwargs: Optional[Dict[str, Any]],
# ) -> None:
#     if mode == "file":
#         # exec(f"source_text = pd.read_{source_file_type}({source_text})")
#         if source_file_type == "csv":
#             source_text = pd.read_csv(source_text)
#         elif source_file_type == "parquet":
#             source_text = pd.read_parquet(source_text)
#         elif source_file_type == "json":
#             source_text = pd.read_json(source_text)
#         elif source_file_type == "sql":
#             source_text = pd.read_sql(source_text)
#         elif source_file_type == "xml":
#             source_text = pd.read_xml(source_text)
#         mode = "pandas"

#     source_text["text"] = source_text["text"].apply(preprocess_sentence)
#     model = TextClassifier(corpus=source_text)
#     model.train_vectorizer()
#     source_text["text"] = model.vectorize(
#         source_text["text"]
#     )  # .apply(model.vectorize)
#     if mode in ["numpy", "pandas"]:
#         (
#             train_sentences,
#             test_sentences,
#             train_sentiments,
#             test_sentiments,
#         ) = train_test_split(
#             source_text["text"],
#             source_text["airline_sentiment"],
#             test_size=kwargs.get("test_size", 0.2),
#             random_state=kwargs.get("random_state", 42),
#         )

#         model.train(train_sentences, train_sentiments)
#         model.test(test_sentences, test_sentiments)
#         model.save_model()
#         print(model.get_report())

#     if mode not in ["csv", "parquet", "json", "sql", "xml"]:
#         raise ValueError("Invalid mode. Valid modes are: pandas, numpy, file")


# if __name__ == "__main__":
#     import sys

#     # sys.path.append(
#     #     "/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/backend"
#     # )
#     train(
#         source_text="/media/anuran/Samsung SSD 970 EVO 1TB/Internship/TrueFoundry/Internship Task/data/airline_sentiment_analysis.csv",
#         mode="file",
#         source_file_type="csv",
#     )
