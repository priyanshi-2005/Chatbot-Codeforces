"""Generate grounded answers from retrieved Codeforces context."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from embeddings import choose_device


DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class AnswerGenerator:
    """Use a small instruction-following model as the generation stage of RAG."""

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATOR_MODEL,
        device: str | None = None,
        max_input_length: int = 2048,
        max_new_tokens: int = 220,
    ) -> None:
        self.model_name = model_name
        self.device = device or choose_device()
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        model_path = self._resolve_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """Use an existing cache snapshot directly, with online fallback."""
        if Path(model_name).exists():
            return model_name
        try:
            return snapshot_download(model_name, local_files_only=True)
        except LocalEntryNotFoundError:
            return model_name

    def build_prompt(
        self,
        question: str,
        context: str,
        history: Sequence[dict[str, str]] = (),
    ) -> str:
        """Build a chat prompt containing the question, history, and sources."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Codeforces tutor. Answer using only the retrieved "
                    "context. For solution questions, prioritize the editorial. Explain "
                    "clearly and concisely without mentioning source labels. If the answer "
                    "is absent, say the indexed data is insufficient. When summarizing a "
                    "problem statement, preserve the transformation, allowed operation, "
                    "and requested output. Do not write or debug code."
                ),
            },
            *history[-6:],
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Question:\n{question.strip()}"
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        question: str,
        context: str,
        history: Sequence[dict[str, str]] = (),
    ) -> str:
        """Generate one answer grounded in the supplied retrieved context."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if not context.strip():
            return "I could not find relevant information in the indexed problems."

        prompt = self.build_prompt(question, context, history)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer or "I could not generate an answer from the retrieved context."
