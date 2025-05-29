import os
import uuid

import chromadb
from chromadb.api.models import Collection
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()


def load_document(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def create_embeddings(strings: list[str]) -> list[float]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(strings)
    return embeddings.tolist()


def save_embeddings(collection: Collection, documents: list[str], embeddings: list[float]) -> None:
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[uuid.uuid4().hex for _ in range(len(documents))],
    )


def build_query_content(query: str, context: list[str]) -> str:
    return f"""
        Question:
        {query}
        Context: 
        {" ".join(context)}
        
        Using the context provided, answer the question as accurately as possible. 
    """


def main() -> None:
    print("-" * 5, end="")
    print("CONFIGURING APPLICATION", end="")
    print("-" * 5)

    API_KEY = os.environ.get("GOGGLE_API_KEY", None)
    if not API_KEY:
        print("GOGGLE_API_KEY environment variable not set")
        return

    DOCUMENT_PATH = os.environ.get("DOCUMENT_PATH", None)
    if not DOCUMENT_PATH:
        print("DOCUMENT_PATH environment variable not set")
        return

    genai_client = genai.Client(api_key=API_KEY)

    print("Creating ChromaDB collection...", end="", flush=True)
    # chromadb_client = chromadb.PersistentClient(path="./chromadb")
    chromadb_client = chromadb.Client()
    chromadb_collection = chromadb_client.get_or_create_collection(name="rag_demo")
    print("✅")

    # SET UP
    # 1. Load document and split using LangChain
    # 2. Create embeddings for the split document using Gemini API
    # 3. Create ChromaDB collection and add embeddings and document splits with ids

    print("Loading documents... ", end="", flush=True)
    documents = load_document(DOCUMENT_PATH)
    print(f"loaded {len(documents)} chucks ✅")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        add_start_index=True
    )
    document_splits = text_splitter.split_documents(documents)

    print("Embedding documents... ", end="", flush=True)
    document_strings = [split.page_content for split in document_splits]
    embeddings = create_embeddings(strings=document_strings)
    print(f"embedded {len(embeddings)} documents ✅")

    print("Saving embeddings...", end="", flush=True)
    save_embeddings(collection=chromadb_collection, documents=document_strings, embeddings=embeddings)
    print("✅")

    print("-" * 5, end="")
    print("APPLICATION READY FOR INPUT", end="")
    print("-" * 5, end="\n\n")

    # RUNTIME
    # 1. Get query from terminal
    # 2. Get embeddings of query
    # 3. Query ChromaDB for embeddings
    # 4. Send to Gemini API for response

    while True:
        query = input("> ")
        if query == "exit":
            break

        query_embeddings = create_embeddings(strings=[query])
        query_result = chromadb_collection.query(query_embeddings=query_embeddings)
        contents = build_query_content(query=query, context=query_result["documents"][0])

        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction="You are a reviewing a document and answering queries based off the reviewed document"
            )
        )
        print(f"{response.text}\n")


if __name__ == "__main__":
    main()
