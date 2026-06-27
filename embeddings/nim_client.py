import os
import requests
from dotenv import load_dotenv

load_dotenv()

NIM_API_KEY = os.getenv("NIM_API_KEY")
EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"  # good general-purpose embedding model on NIM


def get_embedding(text: str, input_type: str = "passage"):
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "input": [text],
        "model": EMBED_MODEL,
        "input_type": input_type,  # "passage" for documents, "query" for search queries
        "encoding_format": "float",
    }
    response = requests.post(EMBED_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


def get_embeddings_batch(texts: list[str], input_type: str = "passage"):
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "input": texts,
        "model": EMBED_MODEL,
        "input_type": input_type,
        "encoding_format": "float",
    }
    response = requests.post(EMBED_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]