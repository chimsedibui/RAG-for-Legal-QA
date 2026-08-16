"""EmbeddingProvider implementation backed by any OpenAI-compatible
embeddings endpoint. Shared by the API (services/search.py) and the offline
pipeline (pipeline/chunk_embedding.py) so there's one place that builds an
embedding client instead of two independent copies."""
from typing import List

from openai import OpenAI


class OpenAIEmbeddingProvider:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model_name

    def embed(self, text: str) -> List[float]:
        """Lấy vector embedding từ model."""
        try:
            response = self._client.embeddings.create(model=self._model_name, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"Lỗi embedding: {e}")
            return []
