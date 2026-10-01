"""Run the conversational Codeforces RAG chatbot in the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from generator import AnswerGenerator, DEFAULT_GENERATOR_MODEL
from retriever import DEFAULT_INDEX_DIRECTORY, ProblemRetriever


class CodeforcesChatbot:
    """Connect retrieval, answer generation, sources, and conversation history."""

    def __init__(
        self,
        index_directory: Path = DEFAULT_INDEX_DIRECTORY,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        device: str | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.retriever = ProblemRetriever(index_directory, device=device)
        self.generator = AnswerGenerator(generator_model, device=device)
        self.top_k = top_k
        self.history: list[dict[str, str]] = []

    def ask(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve context, generate an answer, and remember the conversation."""
        if not question.strip():
            raise ValueError("question must not be empty")

        retrieval_query = question
        if self.history and not self.retriever.has_explicit_problem_reference(question):
            previous_question = next(
                message["content"]
                for message in reversed(self.history)
                if message["role"] == "user"
            )
            retrieval_query = f"{previous_question}\nFollow-up: {question}"

        results = self.retriever.retrieve(retrieval_query, top_k=self.top_k)
        context = self.retriever.format_context(results)
        answer = self.generator.generate(question, context, self.history)

        self.history.extend(
            [
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer, results

    def clear_history(self) -> None:
        """Forget previous turns while keeping the loaded models and index."""
        self.history.clear()


def print_sources(results: list[dict[str, Any]]) -> None:
    """Print compact, de-duplicated source labels after each answer."""
    print("\nSources:")
    seen: set[tuple[str, str]] = set()

    for result in results:
        source_key = (str(result["problem_id"]), str(result["section"]))
        if source_key in seen:
            continue
        seen.add(source_key)
        print(
            f"- {result['problem_id']} - {result['title']} "
            f"({result['section']}): {result['source_url']}"
        )


def run_chat(chatbot: CodeforcesChatbot) -> None:
    """Keep accepting questions until the user exits the terminal session."""
    print("Codeforces RAG Chatbot")
    print("Ask about an indexed problem. Commands: clear, exit")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "clear":
            chatbot.clear_history()
            print("Conversation history cleared.")
            continue

        try:
            answer, results = chatbot.ask(question)
        except Exception as error:
            print(f"Could not answer: {error}")
            continue

        print(f"\nAssistant: {answer}")
        print_sources(results)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    chatbot = CodeforcesChatbot(
        index_directory=args.index_dir,
        generator_model=args.generator_model,
        device=args.device,
        top_k=args.top_k,
    )
    run_chat(chatbot)


if __name__ == "__main__":
    main()
