"""Build, save and search a FAISS index for Codeforces content."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import CodeBERTEmbedder

import faiss


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "problems"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_FILENAME = "codeforces.faiss"
DOCUMENTS_FILENAME = "documents.json"


@dataclass
class DocumentChunk:
    """One searchable piece of a problem statement, editorial or solution."""

    chunk_id: str
    problem_id: str
    title: str
    section: str
    text: str
    source_url: str
    tags: list[str]
    rating: int | None


def split_text(
    text: str,
    max_words: int = 220,
    overlap_words: int = 30,
) -> list[str]:
    """Split long text into overlapping chunks that fit CodeBERT comfortably."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max_words - overlap_words

    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break

    return chunks


def format_examples(examples: list[dict[str, str]]) -> str:
    """Turn structured sample tests into readable retrieval text."""
    formatted = []
    for number, example in enumerate(examples, start=1):
        formatted.append(
            f"Example {number} input:\n{example.get('input', '')}\n"
            f"Example {number} output:\n{example.get('output', '')}"
        )
    return "\n\n".join(formatted)


def create_chunks_for_problem(
    problem: dict[str, Any],
    max_words: int = 220,
    overlap_words: int = 30,
) -> list[DocumentChunk]:
    """Create searchable chunks from every useful part of one problem."""
    problem_id = str(problem["problem_id"])
    title = str(problem["title"])
    tags = [str(tag) for tag in problem.get("tags", [])]
    rating = problem.get("rating")
    problem_url = str(problem.get("url", ""))

    statement_parts = [
        f"Description:\n{problem.get('description', '')}",
        f"Input:\n{problem.get('input', '')}",
        f"Output:\n{problem.get('output', '')}",
        format_examples(problem.get("examples", [])),
        f"Notes:\n{problem.get('note', '')}" if problem.get("note") else "",
    ]
    statement_text = "\n\n".join(part for part in statement_parts if part.strip())

    sections: list[tuple[str, str, str]] = [
        (
            "overview",
            f"Problem title: {title}. "
            f"Problem ID: {problem_id}. "
            f"Tags: {', '.join(tags) if tags else 'not listed'}. "
            f"Rating: {rating if rating is not None else 'not published'}.",
            problem_url,
        ),
        ("statement", statement_text, problem_url)
    ]

    editorial = problem.get("editorial", {})
    editorial_url = str(editorial.get("url") or problem_url)
    editorial_content = str(editorial.get("content", "")).strip()
    if editorial_content:
        sections.append(
            (
                "editorial",
                f"Editorial for {problem_id} - {title}:\n{editorial_content}",
                editorial_url,
            )
        )

    for code_number, code_block in enumerate(editorial.get("code_blocks", []), start=1):
        code_text = str(code_block).strip()
        if code_text:
            sections.append(
                (
                    "solution_code",
                    f"Solution code {code_number} for {problem_id} - {title}:\n{code_text}",
                    editorial_url,
                )
            )

    documents = []
    for section, text, source_url in sections:
        text_chunks = split_text(text, max_words=max_words, overlap_words=overlap_words)
        for chunk_number, chunk_text in enumerate(text_chunks):
            metadata_prefix = (
                f"Problem {problem_id}: {title}\n"
                f"Section: {section}\n"
                f"Tags: {', '.join(tags) if tags else 'not listed'}"
            )
            documents.append(
                DocumentChunk(
                    chunk_id=f"{problem_id}:{section}:{chunk_number}",
                    problem_id=problem_id,
                    title=title,
                    section=section,
                    text=f"{metadata_prefix}\n\n{chunk_text}",
                    source_url=source_url,
                    tags=tags,
                    rating=rating,
                )
            )

    return documents


def load_document_chunks(
    data_directory: Path,
    max_words: int = 220,
    overlap_words: int = 30,
    problem_limit: int | None = None,
) -> list[DocumentChunk]:
    """Load problem JSON files and convert them into searchable chunks."""
    paths = sorted(data_directory.glob("*.json"))
    if problem_limit is not None:
        if problem_limit < 1:
            raise ValueError("problem_limit must be at least 1")
        paths = paths[:problem_limit]

    if not paths:
        raise ValueError(f"No problem JSON files found in {data_directory}")

    documents = []
    for path in paths:
        with path.open(encoding="utf-8") as problem_file:
            problem = json.load(problem_file)
        documents.extend(
            create_chunks_for_problem(
                problem,
                max_words=max_words,
                overlap_words=overlap_words,
            )
        )

    return documents


class VectorStore:
    """Keep normalized embeddings and document metadata in matching order."""

    def __init__(self, dimension: int, embedding_model: str) -> None:
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[DocumentChunk] = []

    def add(self, embeddings: np.ndarray, documents: list[DocumentChunk]) -> None:
        """Add embeddings and their corresponding documents to the index."""
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings shaped (n, {self.dimension}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(documents):
            raise ValueError("The embedding and document counts do not match")

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the highest-scoring documents for one normalized query vector."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected a query vector of size {self.dimension}, got {query.shape[1]}"
            )

        result_count = min(top_k, self.index.ntotal)
        scores, positions = self.index.search(query, result_count)
        results = []

        for score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(self.documents):
                continue
            result = asdict(self.documents[position])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, index_directory: Path) -> None:
        """Persist the FAISS index and its ordered document manifest."""
        index_directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_directory / INDEX_FILENAME))

        manifest = {
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "document_count": len(self.documents),
            "documents": [asdict(document) for document in self.documents],
        }
        with (index_directory / DOCUMENTS_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_directory: Path) -> "VectorStore":
        """Load a saved index and verify that its metadata still matches."""
        index = faiss.read_index(str(index_directory / INDEX_FILENAME))
        with (index_directory / DOCUMENTS_FILENAME).open(encoding="utf-8") as file:
            manifest = json.load(file)

        store = cls(
            dimension=int(manifest["dimension"]),
            embedding_model=str(manifest["embedding_model"]),
        )
        store.index = index
        store.documents = [
            DocumentChunk(**document) for document in manifest["documents"]
        ]

        if store.index.ntotal != len(store.documents):
            raise ValueError("The saved FAISS index and document manifest do not match")
        return store


def build_vector_store(
    data_directory: Path,
    index_directory: Path,
    batch_size: int = 8,
    max_words: int = 220,
    overlap_words: int = 30,
    problem_limit: int | None = None,
) -> VectorStore:
    """Create embeddings for all chunks and save a complete vector store."""
    documents = load_document_chunks(
        data_directory,
        max_words=max_words,
        overlap_words=overlap_words,
        problem_limit=problem_limit,
    )
    print(f"Created {len(documents)} chunks")

    embedder = CodeBERTEmbedder()
    embeddings = embedder.batch_generate_embeddings(
        [document.text for document in documents],
        batch_size=batch_size,
    )

    store = VectorStore(
        dimension=embedder.embedding_dimension,
        embedding_model=embedder.model_name,
    )
    store.add(embeddings, documents)
    store.save(index_directory)
    print(f"Saved {store.index.ntotal} vectors to {index_directory}")
    return store


def parse_arguments() -> argparse.Namespace:
    """Read simple index-building options."""
    parser = argparse.ArgumentParser(description="Build the Codeforces FAISS index")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Build and persist the vector store from command-line options."""
    arguments = parse_arguments()
    build_vector_store(
        data_directory=arguments.data_dir,
        index_directory=arguments.index_dir,
        batch_size=arguments.batch_size,
        max_words=arguments.max_words,
        overlap_words=arguments.overlap_words,
        problem_limit=arguments.limit,
    )


if __name__ == "__main__":
    main()
