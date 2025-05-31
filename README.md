# Retrieval Augmented Generation (RAG) Demo

A simple CLI application to demonstrate Retrieval Augmented Generation (RAG), a technique that enables large language 
models to retrieve and incorporate new information.

The demo application uses:
* [LangChain](https://python.langchain.com/docs/tutorials/rag/#splitting-documents) for splitting documents
* [ChromaDB](https://docs.trychroma.com/docs/overview/introduction) as the vector database
* [Sentence Transformer](https://www.sbert.net/docs/quickstart.html) for document embedding
* [Gemini API](https://ai.google.dev/gemini-api/docs/quickstart) as the Large Language Model (LLM)

## Requirements
1. Python 3
2. UV
3. Gemini API Key

## Setup

1. Make a copy of `.env.template`, renaming the copy to `.env`.
2. Get a [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) and set its value for `GOOGLE_API_KEY` in the `.env` file
3. Set the value of the path to your document for `DOCUMENT_PATH` environment variable.
4. In the root directory create a virtual environment `python3 -m venv .venv` or `uv venv`.

## Running the application

* `uv run main.py` or `python3 main.py`
