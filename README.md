# Codeforces RAG Chatbot

A local retrieval-augmented generation (RAG) chatbot that scrapes Codeforces
problems and editorials, retrieves relevant context with CodeBERT and FAISS,
and generates grounded explanations with a small instruction-tuned model.

This repository combines the scraper and chatbot assignments into one simple
root-level project.

## Project flow

```text
Codeforces
    |
    v
scraper.py -> structured problem/editorial JSON files
    |
    v
vector_store.py -> text chunks -> CodeBERT embeddings -> FAISS index
    |
    v
user question -> retriever.py -> relevant context
    |
    v
generator.py -> grounded answer
    |
    v
chatbot.py -> sources + conversation history
```

## Features

- Scrapes problem statements, metadata, examples, editorials, and solution code.
- Supports both BeautifulSoup HTTP scraping and headless Selenium.
- Preserves mathematical text and applies polite delays between requests.
- Resumes safely by skipping JSON files that already exist.
- Embeds statement, editorial, and solution chunks with CodeBERT.
- Stores normalized 768-dimensional vectors in a FAISS `IndexFlatIP` index.
- Combines semantic similarity with transparent title and problem-ID matching.
- Generates local answers with `Qwen/Qwen2.5-0.5B-Instruct`; no API key is needed.
- Maintains recent conversation history and shows the retrieved sources.

## Repository structure

```text
ChatbotGDG/
|-- scraper.py          # Codeforces problem and editorial scraper
|-- embeddings.py       # CodeBERT single and batch embeddings
|-- vector_store.py     # Chunking, FAISS indexing, save/load
|-- retriever.py        # Semantic retrieval and metadata reranking
|-- generator.py        # Grounded local response generation
|-- chatbot.py          # Terminal chat flow and conversation history
|-- examples.ipynb      # Assignment 2 usage example
|-- requirements.txt    # Python dependencies
|-- data/
|   |-- problems/       # 50 sample problem JSON files
|   `-- index/          # Generated FAISS files (ignored by Git)
|-- Assignment-1.pdf
`-- Assignment_2.pdf
```

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The first embedding or chatbot run downloads CodeBERT and the local Qwen model
to the Hugging Face cache. These model files are not stored in the repository.

### macOS OpenMP note

Some macOS Python environments load different OpenMP runtimes for PyTorch and
FAISS. If Python reports a duplicate OpenMP runtime, prefix commands that use
both packages with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 chatbot.py --device cpu
```

Use the same prefix with `vector_store.py` or `retriever.py` if required by
your environment.

## 1. Scrape problems and editorials

The repository includes 50 sample records. To collect up to 50 current
problems from the first problem-set page:

```bash
python3 scraper.py \
  --browser selenium \
  --start-page 1 \
  --pages 1 \
  --max-problems 50
```

Use `--browser http` for the faster BeautifulSoup-only route. Selenium mode
falls back to HTTP when a page does not load correctly. Existing files are
skipped automatically; add `--overwrite` only when you intentionally want to
replace them.

Each `data/problems/<problem_id>.json` record contains:

- title, ID, contest information, URL, tags, and rating;
- time and memory limits;
- description, input, output, notes, and examples;
- editorial text and extracted solution-code blocks.

The number 50 is a configurable demonstration size, not a hardcoded limit.
It keeps scraping, indexing, and interview demonstrations quick while still
showing the complete pipeline. Increase `--pages` and `--max-problems` to scale
the dataset.

## 2. Build the FAISS index

```bash
python3 vector_store.py \
  --data-dir data/problems \
  --index-dir data/index \
  --batch-size 4
```

The current 50-problem dataset produces 432 overlapping chunks across overview,
statement, editorial, and solution-code sections. The command writes:

- `data/index/codeforces.faiss`
- `data/index/documents.json`

These generated files are ignored by Git because they can be rebuilt from the
tracked JSON dataset.

## 3. Test retrieval

```bash
python3 retriever.py "Explain the solution for Even Simple Path" --top-k 3
```

FAISS supplies semantic candidates. A small metadata reranking step prioritizes
an explicitly written problem title or ID and favors the requested section,
such as an editorial for a solution question.

## 4. Run the chatbot

```bash
python3 chatbot.py --device cpu --top-k 3
```

Example:

```text
You: Explain the main idea for solving Even Simple Path.
Assistant: The main idea is to encode the degree pattern of a simple path as a matching...

You: How is that matching constructed?
Assistant: Split each vertex into left and right copies...
```

Commands inside the chat:

- `clear` removes the current conversation history.
- `exit` or `quit` closes the chatbot.

The chatbot is designed for explanations based on the indexed data. It does not
write or debug competitive-programming code.

## Example notebook

Open `examples.ipynb` directly in VS Code or Jupyter after building the index.
It demonstrates single and batch CodeBERT embeddings, FAISS retrieval, grounded
answer generation, displayed sources, and a history-aware follow-up.

## Implementation notes

- CodeBERT uses attention-mask-aware mean pooling so padding tokens do not alter
  the embedding.
- Vectors are L2-normalized, allowing `IndexFlatIP` to behave like cosine
  similarity search.
- Long fields are split into overlapping word chunks so editorials and problem
  statements, rather than only titles, are searchable.
- The saved document manifest keeps every FAISS position aligned with its source
  problem, section, URL, tags, and rating.
- The local generator receives only retrieved context plus recent conversation
  turns and reports sources after every answer.

## Validation completed

- 50 structured problem records load successfully.
- The saved FAISS index contains 432 vectors with 768 dimensions.
- Named-title and problem-ID searches return the intended problem first.
- Statement, solution-explanation, topic-switch, and follow-up chat flows run.
- The complete pipeline works locally without an external API key.

## Limitations

- Codeforces HTML can change, so scraper selectors may eventually need updates.
- Scraping speed is intentionally limited to avoid overloading Codeforces.
- The compact local generator is suitable for demonstrations, but retrieved
  sources should still be checked for high-stakes or highly technical answers.
