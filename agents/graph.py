"""
LangGraph multi-agent supervisor graph.
Routes queries through supervisor -> specialist agents -> synthesizer.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.chitchat_agent import chitchat_node
from agents.knowledge_agent import knowledge_agent_node
from agents.sql_agent import sql_agent_node
from agents.supervisor import supervisor_node
from utils import setup_logging

logger = setup_logging("graph")


# ---------------------------------------------------------------------------
# Custom reducers
# ---------------------------------------------------------------------------

def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, preserving keys from both. Used for agent_outputs."""
    merged = left.copy()
    merged.update(right)
    return merged


def add_messages(left: list, right: list) -> list:
    """Append new messages to existing list."""
    return left + right


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    final_response: str
    agent_outputs: Annotated[dict, merge_dicts]
    last_uploaded_doc: str  # Filename of the most recently uploaded document
    final_prompt: list  # Prompt messages ready for streaming (built by synthesizer/chitchat)
    sql_enabled: bool  # Whether the SQL database source is enabled
    exclude_sources: list  # Source filenames to exclude from knowledge base searches


# ---------------------------------------------------------------------------
# Synthesizer node
# ---------------------------------------------------------------------------

SQL_SUMMARIZE_SYSTEM = """You are a helpful customer support data analyst. Based on the user's question and the database query results below, provide a clear, natural language summary.

Database Query Results:
{sql_result}

Provide a helpful, concise summary. If the results are empty, say so clearly."""

KB_ANSWER_SYSTEM = """You are a knowledgeable customer support assistant for TechCorp Solutions. Answer the user's question based ONLY on the provided context from our company documents.

Rules:
1. Only use information from the provided context to answer.
2. If the context doesn't contain enough information, say so clearly.
3. Always cite your sources by mentioning the document name and page number.
   Example: "According to company_refund_policy.pdf (page 2), ..."
4. Be concise but thorough in your answers.
5. Use bullet points for lists when appropriate.

Context from company documents:
{kb_context}"""

BOTH_MERGE_SYSTEM = """You are a customer support assistant. Combine the following database results and policy information into a single coherent response that answers the user's question. Cite sources where applicable.

Database Results:
{sql_result}

Policy/Document Context:
{kb_result}

Provide a unified, helpful response."""


def _get_user_query(state: dict) -> str:
    """Extract the last user message from state."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def synthesizer_node(state: dict) -> dict:
    """
    Synthesizer node: assembles the final prompt for streaming.
    Does NOT call the LLM — returns final_prompt for token-by-token streaming in app.py.
    """
    route = state.get("next_agent", "")
    agent_outputs = state.get("agent_outputs", {})
    user_query = _get_user_query(state)

    sql_result = agent_outputs.get("sql", "")
    kb_result = agent_outputs.get("knowledge", "")

    if route == "both" and sql_result and kb_result:
        prompt_messages = [
            SystemMessage(content=BOTH_MERGE_SYSTEM.format(
                sql_result=sql_result,
                kb_result=kb_result,
            )),
            HumanMessage(content=user_query or "Please combine the above information."),
        ]
    elif route == "sql_agent" and sql_result:
        prompt_messages = [
            SystemMessage(content=SQL_SUMMARIZE_SYSTEM.format(sql_result=sql_result)),
            HumanMessage(content=user_query),
        ]
    elif route == "knowledge_agent" and kb_result:
        prompt_messages = [
            SystemMessage(content=KB_ANSWER_SYSTEM.format(kb_context=kb_result)),
            HumanMessage(content=user_query),
        ]
    else:
        # Edge case: no data retrieved — build a simple prompt
        prompt_messages = [
            SystemMessage(content="You are a helpful customer support assistant."),
            HumanMessage(content=f"I wasn't able to find relevant information for: {user_query}. "
                         "Let the user know and suggest what they can ask about."),
        ]

    return {
        "final_prompt": prompt_messages,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_from_supervisor(state: AgentState) -> str:
    """Routes from supervisor. Maps 'both' to sql_agent (first step of dual path)."""
    route = state.get("next_agent", "chitchat")
    if route == "both":
        return "sql_agent"
    return route


def route_after_sql(state: AgentState) -> str:
    """After SQL agent: chain to knowledge_agent if route is 'both', else synthesizer."""
    if state.get("next_agent") == "both":
        return "knowledge_agent"
    return "synthesizer"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the LangGraph multi-agent supervisor graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("knowledge_agent", knowledge_agent_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor -> conditional routing
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "sql_agent": "sql_agent",
        "knowledge_agent": "knowledge_agent",
        "chitchat": "chitchat",
    })

    # SQL agent -> conditional (synthesizer or knowledge_agent if "both")
    graph.add_conditional_edges("sql_agent", route_after_sql, {
        "knowledge_agent": "knowledge_agent",
        "synthesizer": "synthesizer",
    })

    # Knowledge agent -> always synthesizer
    graph.add_edge("knowledge_agent", "synthesizer")

    # Terminal edges
    graph.add_edge("synthesizer", END)
    graph.add_edge("chitchat", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("Multi-agent graph compiled successfully with MemorySaver checkpointer.")
    return compiled


def get_graph_image(compiled_graph) -> bytes | None:
    """Export the graph as a PNG image (requires graphviz). Returns bytes or None."""
    try:
        return compiled_graph.get_graph().draw_mermaid_png()
    except Exception:
        return None
