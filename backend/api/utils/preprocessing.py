import pandas as pd
import numpy as np
import re
from typing import Optional, List, Any, Dict


def remove_abbreviations(sentence: str) -> str:
    abb_dict: Dict[str, str] = {
        "i'm": "i am",
        "i've": "i have",
        "i'd": "i would",
        "i'll": "i will",
        "he's": "he is",
        "he'd": "he would",
        "he'll": "he will",
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd've": "he would have",
        "he'll've": "he will have",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd've": "i would have",
        "i'll've": "i will have",
        "i'm'a": "i am about to",
        "i'm'o": "i am going to",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you're": "you are",
        "you've": "you have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "'em": "them",
        "wanna": "want to",
        "gonna": "going to",
        "gotta": "got to",
        "lemme": "let me",
        "more'n": "more than",
        "'bout": "about",
        "'til": "until",
        "kinda": "kind of",
        "sorta": "sort of",
        "lotta": "lot of",
        "aught": "ought",
        "methinks": "me thinks",
        "methinks": "me thinks",
        "o'er": "over",
        "tis": "it is",
        "tisn't": "it is not",
        "twas": "it was",
        "twasn't": "it was not",
        "wot": "what",
        "wotcha": "what are you",
        "it's": "it is",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i've": "i have",
    }

    for abb, full in abb_dict.items():
        sentence = sentence.replace(abb, full)

    return sentence


def preprocess_sentence(
    sentence: str, *args: Optional[List[Any]], **kwargs: Optional[Dict[str, Any]]
) -> None:
    sentence = sentence.lower()
    sentence = remove_abbreviations(sentence)
    sentence = re.sub(r"\W", " ", str(sentence))

    # remove all single characters
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Remove single characters from the start
    sentence = re.sub(r"\^[a-zA-Z]\s+", " ", sentence)

    # Substituting multiple spaces with single space
    sentence = re.sub(r"\s+", " ", sentence, flags=re.I)

    # Removing prefixed 'b'
    sentence = re.sub(r"^b\s+", "", sentence)

    return sentence
