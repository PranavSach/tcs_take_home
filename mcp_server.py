"""
FastMCP server exposing customer support AI tools via Model Context Protocol.
Run standalone: python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

from tools.kb_tools import initialize_vector_store, search_knowledge_base, upload_document
from tools.sql_tools import query_database

mcp = FastMCP("Customer Support AI")

# Ensure vector store is initialized on startup
initialize_vector_store()


@mcp.tool()
def query_customer_data(query: str) -> str:
    """Query the customer database using natural language. Use for questions about customers, tickets, interactions, and support data."""
    return query_database(query)


@mcp.tool()
def search_policies(query: str) -> str:
    """Search company policy documents and knowledge base. Use for questions about refund policies, terms of service, and company rules."""
    results = search_knowledge_base(query)
    if not results:
        return "No relevant policy documents found."

    formatted_parts = []
    for r in results:
        formatted_parts.append(
            f"[{r['source']}, Page {r['page']}]\n{r['content']}"
        )
    return "\n\n---\n\n".join(formatted_parts)


@mcp.tool()
def upload_policy_document(file_path: str) -> str:
    """Upload and index a new PDF document into the knowledge base."""
    import os
    filename = os.path.basename(file_path)
    try:
        chunks_added = upload_document(file_path, filename)
        return f"Successfully uploaded '{filename}' and indexed {chunks_added} text chunks into the knowledge base."
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Failed to upload document: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
