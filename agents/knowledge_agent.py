"""
Knowledge specialist agent: RAG-based retrieval over policy documents using ChromaDB.
Returns retrieved context for downstream streaming — does NOT call the LLM.
"""

import re

from langchain_core.messages import AIMessage

from tools.kb_tools import search_knowledge_base
from utils import setup_logging

logger = setup_logging("knowledge_agent")

# Patterns that indicate a vague reference to the most recently uploaded document
VAGUE_DOC_PATTERN = re.compile(
    r"\b(this pdf|the pdf|that pdf|this document|the document|that document|"
    r"uploaded document|uploaded file|the uploaded|this file|that file|the file)\b",
    re.IGNORECASE,
)


def _resolve_doc_references(query: str, last_uploaded_doc: str) -> str:
    """
    If the query contains vague references like 'this pdf' or 'the document',
    prepend the filename context so ChromaDB retrieves the right chunks.
    E.g. "tell me about this pdf" -> "summary of PreQualification Instructions AI ML developers: tell me about this pdf"
    """
    if not last_uploaded_doc or not VAGUE_DOC_PATTERN.search(query):
        return query

    # Strip extension to get a human-readable document name
    doc_name = re.sub(r"\.[^.]+$", "", last_uploaded_doc)
    # Replace underscores/hyphens with spaces for better embedding match
    doc_name = doc_name.replace("_", " ").replace("-", " ")

    enriched = f"summary of {doc_name}: {query}"
    logger.info(f"Resolved vague doc reference: '{query}' -> '{enriched}'")
    return enriched


def knowledge_agent_node(state: dict) -> dict:
    """
    Knowledge agent node: retrieves relevant documents and returns formatted context.
    Does NOT call the LLM — the final generation is streamed in app.py.
    Updates agent_outputs with {"knowledge": context_string}.
    """
    messages = state.get("messages", [])
    last_uploaded_doc = state.get("last_uploaded_doc", "")

    # Get the last user message
    user_query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_query = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        error_msg = "I couldn't find a question to search the knowledge base with."
        return {
            "agent_outputs": {"knowledge": error_msg},
            "messages": [AIMessage(content=error_msg)],
        }

    try:
        # Resolve vague document references using the last uploaded filename
        search_query = _resolve_doc_references(user_query, last_uploaded_doc)

        # If the user is referring to a specific uploaded doc, filter search to that source
        source_filter = None
        if last_uploaded_doc and VAGUE_DOC_PATTERN.search(user_query):
            source_filter = last_uploaded_doc

        # Get exclude_sources from state (set by Data Sources toggles in sidebar)
        exclude_sources = state.get("exclude_sources", []) or []

        # Retrieve relevant documents
        logger.info(f"Searching knowledge base: {search_query[:100]}")
        results = search_knowledge_base(
            search_query,
            source_filter=source_filter,
            exclude_sources=exclude_sources if not source_filter else None,
        )

        if not results:
            no_results_msg = (
                "I couldn't find relevant information in our knowledge base. "
                "Please make sure policy documents have been uploaded and indexed."
            )
            return {
                "agent_outputs": {"knowledge": no_results_msg},
                "messages": [AIMessage(content=no_results_msg)],
            }

        # Build formatted context from search results (no LLM call)
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["source"]
            page = result["page"]
            content = result["content"]
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        return {
            "agent_outputs": {"knowledge": context},
            "messages": [AIMessage(content=f"Retrieved {len(results)} relevant document chunks.")],
        }

    except Exception as e:
        error_msg = f"I encountered an error searching the knowledge base: {str(e)}"
        logger.error(error_msg)
        return {
            "agent_outputs": {"knowledge": error_msg},
            "messages": [AIMessage(content=error_msg)],
        }
