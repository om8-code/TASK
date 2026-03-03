# Repository Overview

This repository implements a **FastAPI** service that provides a semantic search and question‑answering interface over two CSV data sources. The core functionality consists of:

1. **Data Ingestion** – CSV files are loaded with **pandas**, each row is transformed into a plain‑text document using a `row_to_text` helper.
2. **Tokenisation & Indexing** – Documents are tokenised with a simple regular‑expression tokenizer and indexed using **BM25** (via the `rank_bm25` library) wrapped in a custom `Retriever` class.
3. **Retrieval API** – A FastAPI endpoint accepts a user query, retrieves the most relevant documents, and returns them as a JSON response.
4. **Optional LLM Augmentation** – When an `OPENROUTER_API_KEY` (or similar) is present, the service can forward the retrieved context to a large language model (LLM) from **Groq**. The system prompt forces the LLM to answer **only** based on the supplied CSV data, falling back to a predefined message when no answer can be found.

The project is deliberately lightweight, relying on a small set of well‑known Python packages (see `requirements.txt`). It is designed to be **container‑friendly** and can be run locally with `uvicorn` or deployed to any platform that supports FastAPI.

---

## Features

- **FastAPI**‑based HTTP API with automatic OpenAPI documentation.
- **Environment‑driven configuration** using `python-dotenv` for secrets and runtime flags.
- **Pydantic** models for request validation and response serialization.
- **BM25 retrieval** for fast, relevance‑based document ranking.
- **LLM integration** (Groq) that is toggled via an environment variable – no LLM calls when the key is absent.
- **Lazy resource initialization** at application startup, ensuring fast cold starts.
- **Docker‑ready** (the repo includes a typical `Dockerfile` pattern in the project root – not shown here).

---

## Architecture Overview

```
+-------------------+        +-------------------+        +-------------------+
|   FastAPI Server  | <----> |   Retriever (BM25)| <----> |   CSV Documents   |
+-------------------+        +-------------------+        +-------------------+
        |                                 |
        | (optional)                      | (optional)
        v                                 v
+-------------------+        +-------------------+
|   Groq LLM API    | <----> |   Prompt Builder  |
+-------------------+        +-------------------+
```

- **FastAPI Server** – Handles HTTP requests, loads configuration, and wires dependencies.
- **Retriever** – Encapsulates BM25 indexing and provides a `search(query, k)` method.
- **Prompt Builder** – Constructs a system prompt that instructs the LLM to answer strictly from the retrieved context.
- **Groq LLM API** – Called only when the `OPENROUTER_API_KEY` environment variable is set.

---

## Getting Started

### Prerequisites

- Python **3.9+**
- `pip` (or `uvicorn` for running the server)
- (Optional) An API key for Groq/OpenRouter if you want LLM support.

### Installation

```bash
# Clone the repository
git clone https://github.com/your‑org/your‑repo.git
cd your‑repo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with the following variables:

```dotenv
# Required – path to the CSV files (comma‑separated if multiple)
CSV_PATHS=data/file1.csv,data/file2.csv

# Optional – enable LLM integration
OPENROUTER_API_KEY=your‑groq‑api‑key   # leave empty to disable LLM calls
```

### Running the Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API documentation will be available at `http://localhost:8000/docs`.

---

## Usage Example

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the average sales price for Q2 2023?"}'
```

**Response (when LLM is enabled):**

```json
{
  "answer": "The average sales price for Q2 2023 is $12,450.",
  "sources": [
    "row 42 from sales_q2_2023.csv",
    "row 57 from sales_q2_2023.csv"
  ]
}
```

If the LLM key is not set, the service returns only the retrieved documents:

```json
{
  "documents": [
    "...",
    "..."
  ]
}
```

---

## Application Scenarios

- **Internal Knowledge Bases** – Quickly expose searchable access to CSV‑based reports, logs, or inventories.
- **Data‑Driven Chatbots** – Combine deterministic BM25 retrieval with LLM generation for conversational interfaces that stay grounded in factual data.
- **Rapid Prototyping** – Developers can spin up a searchable API for any tabular dataset without building a full‑text search engine.
- **Compliance‑Sensitive Environments** – The system prompt forces the LLM to answer only from known documents, reducing the risk of hallucinations.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request following the standard GitHub workflow.
