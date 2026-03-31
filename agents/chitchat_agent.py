"""
Chitchat agent: handles greetings, general conversation, and out-of-scope queries.
Flags the route for downstream streaming — does NOT call the LLM.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils import setup_logging

logger = setup_logging("chitchat_agent")

CHITCHAT_SYSTEM_PROMPT = """You are a friendly and professional customer support assistant for TechCorp Solutions. Your name is TechCorp AI Assistant.

You can help with:
- Querying customer data and support ticket information
- Answering questions about company policies (refund policy, terms of service, SLA)
- Uploading and searching policy documents

For general greetings and conversation, be warm and helpful. Let the user know what you can assist with.

Keep responses concise and professional."""

CHITCHAT_SQL_DISABLED_PROMPT = """You are a friendly and professional customer support assistant for TechCorp Solutions. Your name is TechCorp AI Assistant.

The customer SQL database is currently disabled by the user. You CANNOT query customer data, tickets, or interactions right now.

You can still help with:
- Answering questions about company policies (refund policy, terms of service, SLA)
- Uploading and searching policy documents

If the user asks about customer data, politely let them know the SQL database is currently disabled and suggest they enable it in the sidebar.

Keep responses concise and professional."""


def chitchat_node(state: dict) -> dict:
    """
    Chitchat node: sets agent_outputs flag and builds final_prompt for streaming.
    Does NOT call the LLM — the final generation is streamed in app.py.
    Uses a SQL-disabled variant prompt when the supervisor downgraded the route.
    """
    messages = state.get("messages", [])
    sql_enabled = state.get("sql_enabled", True)

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
        user_query = "Hello"

    # Use the SQL-disabled prompt if SQL was turned off (route was downgraded)
    system_prompt = CHITCHAT_SYSTEM_PROMPT if sql_enabled else CHITCHAT_SQL_DISABLED_PROMPT

    # Build the prompt for streaming (no LLM call here)
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    return {
        "agent_outputs": {"chitchat": True},
        "final_prompt": prompt_messages,
    }
