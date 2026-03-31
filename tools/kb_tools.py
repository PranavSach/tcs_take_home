"""
Knowledge base tools for vector search over PDF documents using ChromaDB.
"""

import os
from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K, get_embeddings
from utils import setup_logging

logger = setup_logging("kb_tools")

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")
COLLECTION_NAME = "policy_documents"


def _get_chroma_client():
    """Get or create a persistent ChromaDB client."""
    persist_dir = os.path.abspath(CHROMA_PERSIST_DIR)
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _load_and_split_pdf(file_path: str) -> list:
    """Load a PDF and split into chunks with metadata."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_documents(pages)
    filename = os.path.basename(file_path)

    for chunk in chunks:
        chunk.metadata["source"] = filename
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0

    return chunks


def initialize_vector_store():
    """
    Initialize the ChromaDB vector store with sample PDFs from documents/.
    Skips re-indexing if documents are already loaded (checks collection count).
    Returns the ChromaDB collection.
    """
    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() > 0:
        logger.info(f"Vector store already initialized with {collection.count()} documents. Skipping.")
        return collection

    # Load all PDFs from documents directory
    docs_path = os.path.abspath(DOCUMENTS_DIR)
    if not os.path.exists(docs_path):
        logger.warning(f"Documents directory not found: {docs_path}")
        return collection

    pdf_files = list(Path(docs_path).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in documents directory.")
        return collection

    embeddings = get_embeddings()
    total_chunks = 0

    for pdf_path in pdf_files:
        try:
            chunks = _load_and_split_pdf(str(pdf_path))
            if not chunks:
                continue

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{"source": chunk.metadata["source"], "page": chunk.metadata.get("page", 0)} for chunk in chunks]
            ids = [f"{pdf_path.stem}_{i}" for i in range(total_chunks, total_chunks + len(chunks))]

            # Embed and add to collection
            embedded = embeddings.embed_documents(texts)
            collection.add(
                ids=ids,
                embeddings=embedded,
                documents=texts,
                metadatas=metadatas,
            )
            total_chunks += len(chunks)
            logger.info(f"Indexed {len(chunks)} chunks from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Vector store initialized with {total_chunks} total chunks.")
    return collection


def upload_document(file_path: str, filename: str, replace: bool = False) -> int | str:
    """
    Upload and index a new PDF into the knowledge base.

    Args:
        file_path: Path to the PDF file on disk.
        filename: Original filename (used as the source metadata).
        replace: If True and the document already exists, delete old chunks first.

    Returns:
        int (chunk count) on success, or a str message if the document was skipped.
    """
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Check for existing chunks with the same source filename
        existing = collection.get(where={"source": filename})
        existing_count = len(existing["ids"]) if existing and existing["ids"] else 0

        if existing_count > 0 and not replace:
            msg = f"Document '{filename}' is already indexed with {existing_count} chunks. Skipping duplicate upload."
            logger.info(msg)
            return msg

        if existing_count > 0 and replace:
            collection.delete(where={"source": filename})
            logger.info(f"Deleted {existing_count} existing chunks for '{filename}' before re-indexing.")

        chunks = _load_and_split_pdf(file_path)
        if not chunks:
            return 0

        embeddings = get_embeddings()
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{"source": filename, "page": chunk.metadata.get("page", 0)} for chunk in chunks]

        # Use filename stem + index for unique IDs
        ids = [f"{Path(filename).stem}_{i}" for i in range(len(chunks))]

        embedded = embeddings.embed_documents(texts)
        collection.add(
            ids=ids,
            embeddings=embedded,
            documents=texts,
            metadatas=metadatas,
        )

        action = "Re-indexed" if replace and existing_count > 0 else "Uploaded"
        logger.info(f"{action} {len(chunks)} chunks from {filename}")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to upload document {filename}: {e}")
        raise


def search_knowledge_base(
    query: str,
    k: int = RETRIEVAL_TOP_K,
    source_filter: str | None = None,
    exclude_sources: list[str] | None = None,
) -> list[dict]:
    """
    Search the knowledge base for documents relevant to the query.
    Returns a list of dicts with 'content', 'source', and 'page' keys.

    Args:
        query: The search query text.
        k: Number of top results to return.
        source_filter: If provided, restrict results to chunks from this source filename.
        exclude_sources: If provided, exclude chunks from these source filenames.
    """
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        if collection.count() == 0:
            logger.warning("Knowledge base is empty. Run setup_database.py first.")
            return []

        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(query)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }

        # source_filter takes precedence over exclude_sources (they're mutually exclusive)
        if source_filter:
            query_kwargs["where"] = {"source": source_filter}
            logger.info(f"Filtering search to source: {source_filter}")
        elif exclude_sources:
            query_kwargs["where"] = {"source": {"$nin": exclude_sources}}
            logger.info(f"Excluding sources from search: {exclude_sources}")

        results = collection.query(**query_kwargs)

        formatted = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                formatted.append({
                    "content": doc,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "N/A"),
                    "distance": dist,
                })

        return formatted

    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return []
