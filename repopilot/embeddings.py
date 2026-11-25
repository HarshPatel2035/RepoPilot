from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> Sequence[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Local ONNX embeddings via fastembed: no GPU and no PyTorch needed in the container."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # heavy import, only when actually used

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._model.embed(texts, batch_size=64))

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the model's query instruction (BGE expects one for questions).
        return next(iter(self._model.query_embed(text)))
