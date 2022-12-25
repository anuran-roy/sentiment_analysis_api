import requests
import os


async def infer(sentence: str) -> str:
    """This function calls the external API to get the sentiment of a sentence."""
    url = "https://twinword-sentiment-analysis.p.rapidapi.com/analyze/"

    querystring = {"text": sentence}

    headers = {
        "X-RapidAPI-Key": os.environ.get("RAPID_API_KEY"),
        "X-RapidAPI-Host": "twinword-sentiment-analysis.p.rapidapi.com",
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    return response.json()["type"]
