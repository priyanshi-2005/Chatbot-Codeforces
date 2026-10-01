"""Generate normalized CodeBERT embeddings for text chunks and queries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "microsoft/codebert-base"


def choose_device() -> str:
    """Use the best available device while always supporting a CPU fallback."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CodeBERTEmbedder:
    """Convert natural-language and code text into searchable vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or choose_device()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def embedding_dimension(self) -> int:
        """Return the vector size produced by the loaded model."""
        return int(self.model.config.hidden_size)

    def _masked_mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average only real tokens, excluding padding from the final vector."""
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        expanded_mask = expanded_mask.to(token_embeddings.dtype)

        summed_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
        token_counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return summed_embeddings / token_counts

    def batch_generate_embeddings(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Generate one normalized float32 embedding for every input text."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}

            with torch.inference_mode():
                model_output = self.model(**encoded)
                pooled = self._masked_mean_pool(
                    model_output.last_hidden_state,
                    encoded["attention_mask"],
                )
                normalized = functional.normalize(pooled, p=2, dim=1)

            embeddings.append(normalized.cpu().numpy().astype(np.float32))

        return np.concatenate(embeddings, axis=0)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate one embedding for a document chunk or user query."""
        if not text.strip():
            raise ValueError("text must not be empty")
        return self.batch_generate_embeddings([text], batch_size=1)[0]
