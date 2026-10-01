"""Retrieve the most relevant Codeforces problem chunks from the FAISS index."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from embeddings import CodeBERTEmbedder
from vector_store import VectorStore


DEFAULT_INDEX_DIRECTORY = Path("data/index")
SOLUTION_WORDS = {"approach", "explain", "solution", "solve"}
STATEMENT_WORDS = {"constraint", "constraints", "input", "output", "statement"}
CODE_WORDS = {"code", "coding", "implement", "implementation", "python"}


def tokenize(text: str) -> set[str]:
    """Return lowercase words used for lightweight metadata matching."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class ProblemRetriever:
    """Combine semantic FAISS search with small, transparent metadata boosts."""

    def __init__(
        self,
        index_directory: Path = DEFAULT_INDEX_DIRECTORY,
        device: str | None = None,
        candidate_pool: int = 50,
    ) -> None:
        if candidate_pool < 1:
            raise ValueError("candidate_pool must be at least 1")

        # Load CodeBERT before FAISS. This import order avoids a macOS OpenMP clash.
        self.embedder = CodeBERTEmbedder(device=device)
        self.store = VectorStore.load(index_directory)
        self.candidate_pool = candidate_pool
        self.problem_titles = {
            document.problem_id: document.title for document in self.store.documents
        }

        if self.embedder.embedding_dimension != self.store.dimension:
            raise ValueError(
                "The embedding model and saved FAISS index have different dimensions"
            )
        if self.embedder.model_name != self.store.embedding_model:
            raise ValueError(
                "The embedding model does not match the model used to build the index"
            )

    def _explicit_problem_ids(self, query: str) -> set[str]:
        """Find problem IDs whose ID or complete title appears in the query."""
        query_lower = query.lower()
        query_words = tokenize(query)
        matches = set()

        for problem_id, title in self.problem_titles.items():
            if problem_id.lower() in query_words or title.lower() in query_lower:
                matches.add(problem_id)

        return matches

    def has_explicit_problem_reference(self, query: str) -> bool:
        """Return whether the query names an indexed problem title or ID."""
        return bool(self._explicit_problem_ids(query))

    def _metadata_bonus(self, query: str, result: dict[str, Any]) -> float:
        """Boost explicit title, problem-ID, and requested-section matches."""
        query_lower = query.lower()
        query_words = tokenize(query)
        title = str(result["title"]).lower().strip()
        title_words = tokenize(title)
        problem_id = str(result["problem_id"]).lower()
        section = str(result["section"])
        text = str(result["text"])
        solution_requested = bool(query_words & SOLUTION_WORDS)
        code_requested = bool(query_words & CODE_WORDS) or "c++" in query_lower

        bonus = 0.0

        if title and title in query_lower:
            bonus += 0.20
        elif title_words:
            title_coverage = len(query_words & title_words) / len(title_words)
            if title_coverage >= 0.5:
                bonus += 0.08 * title_coverage

        if problem_id in query_words:
            bonus += 0.25

        if solution_requested and section == "editorial":
            bonus += 0.05
            if str(result["chunk_id"]).endswith(":editorial:0"):
                bonus += 0.08
        if code_requested and section == "solution_code":
            bonus += 0.08
        if query_words & STATEMENT_WORDS and section == "statement":
            bonus += 0.05
            if str(result["chunk_id"]).endswith(":statement:0"):
                bonus += 0.10
        if not code_requested and (
            section == "solution_code" or self._looks_like_code(text)
        ):
            bonus -= 0.15

        return bonus

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """Identify code-heavy chunks that are unhelpful for explanation queries."""
        code_markers = ("#include", "int main", "std::", "cin >>", "cout <<", "};")
        return sum(marker in text for marker in code_markers) >= 2

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return reranked chunks for a non-empty user query."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self.embedder.generate_embedding(query)
        candidate_count = max(self.candidate_pool, top_k * 10)
        if self._explicit_problem_ids(query):
            # Search the full small index so every chunk of a named problem can
            # receive the title/ID boost, including its detailed editorial.
            candidate_count = self.store.index.ntotal
        candidates = self.store.search(query_embedding, top_k=candidate_count)

        reranked: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for candidate in candidates:
            chunk_id = str(candidate["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            item = candidate.copy()
            item["semantic_score"] = float(item.pop("score"))
            item["metadata_bonus"] = self._metadata_bonus(query, item)
            item["score"] = item["semantic_score"] + item["metadata_bonus"]
            reranked.append(item)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def format_context(results: list[dict[str, Any]]) -> str:
        """Turn retrieved chunks into labeled context for the chatbot."""
        sections = []
        for rank, result in enumerate(results, start=1):
            header = (
                f"[Source {rank}: {result['problem_id']} - {result['title']} "
                f"({result['section']})]"
            )
            sections.append(f"{header}\n{result['text']}")
        return "\n\n".join(sections)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or Codeforces problem to search for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    retriever = ProblemRetriever(args.index_dir, device=args.device)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['problem_id']} - {result['title']} "
            f"[{result['section']}] score={result['score']:.4f}"
        )


if __name__ == "__main__":
    main()
