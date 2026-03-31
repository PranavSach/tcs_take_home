"""
SQL specialist agent: handles queries about customer data, tickets, and interactions.
"""

from langchain_core.messages import AIMessage

from tools.sql_tools import query_database
from utils import setup_logging

logger = setup_logging("sql_agent")


def sql_agent_node(state: dict) -> dict:
    """
    SQL agent node: extracts user query, calls the database tool, returns results.
    Updates agent_outputs with {"sql": result} using the merge_dicts reducer.
    """
    messages = state.get("messages", [])

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
        error_msg = "I couldn't find a question to query the database with."
        return {
            "agent_outputs": {"sql": error_msg},
            "messages": [AIMessage(content=error_msg)],
        }

    try:
        logger.info(f"Querying database: {user_query[:100]}")
        result = query_database(user_query)
        return {
            "agent_outputs": {"sql": result},
            "messages": [AIMessage(content=result)],
        }
    except Exception as e:
        error_msg = f"I encountered an error querying the database: {str(e)}"
        logger.error(error_msg)
        return {
            "agent_outputs": {"sql": error_msg},
            "messages": [AIMessage(content=error_msg)],
        }
