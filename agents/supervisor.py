"""
Supervisor agent: routes user queries to the appropriate specialist agent.
Uses a defensive parsing cascade: exact match -> keyword search -> fallback to chitchat.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from utils import setup_logging

logger = setup_logging("supervisor")

VALID_ROUTES = {"sql_agent", "knowledge_agent", "both", "chitchat"}

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor for a customer support AI system. Your job is to classify the user's query into exactly one category.

Classify the query as:
- sql_agent: Questions about customer data, profiles, emails, phone numbers, tickets, ticket counts, statuses, priorities, interactions, support history, or any database lookups.
- knowledge_agent: Questions about company policies, refund policy, terms of service, SLA, data privacy, acceptable use, or any document/policy content.
- both: Questions that need BOTH customer data AND policy context. Example: "Can customer X get a refund based on our policy?" needs the customer's data AND the refund policy.
- chitchat: Greetings, general conversation, questions not related to customer data or policies.

Examples:
- "What is Ema's email?" -> sql_agent
- "How many open tickets are there?" -> sql_agent
- "Show me all enterprise customers" -> sql_agent
- "Which customers have critical priority tickets?" -> sql_agent
- "What is the refund policy?" -> knowledge_agent
- "What is the SLA uptime guarantee?" -> knowledge_agent
- "What are the terms for account termination?" -> knowledge_agent
- "Tell me about this pdf" -> knowledge_agent
- "What's in the uploaded document?" -> knowledge_agent
- "Summarize the document I just uploaded" -> knowledge_agent
- "What does the document say about refunds?" -> knowledge_agent
- "Can customer Ema get a refund based on our policy?" -> both
- "Is this customer eligible for a refund given their signup date?" -> both
- "Hello, how are you?" -> chitchat
- "What can you help me with?" -> chitchat

Respond with ONLY one of: sql_agent, knowledge_agent, both, chitchat"""

# Keywords in the user query that should bias routing toward knowledge_agent
DOCUMENT_KEYWORDS = re.compile(
    r"\b(pdf|document|uploaded|file|policy|policies)\b", re.IGNORECASE
)


def parse_route(llm_response: str, user_query: str = "") -> str:
    """
    Defensive parsing cascade to extract route from LLM response.
    1. Exact match after stripping
    2. Keyword search in response text
    3. Check user query for document-related keywords -> knowledge_agent
    4. Fallback to chitchat
    """
    cleaned = llm_response.strip().lower().replace("'", "").replace('"', "")

    # Level 1: Exact match
    if cleaned in VALID_ROUTES:
        return cleaned

    # Level 2: Keyword search (order matters — check 'both' before individual agents)
    if "both" in cleaned:
        return "both"
    if "sql_agent" in cleaned:
        return "sql_agent"
    if "knowledge_agent" in cleaned:
        return "knowledge_agent"
    if "chitchat" in cleaned:
        return "chitchat"

    # Level 2b: Partial keyword matching for common LLM variations
    if re.search(r"\bsql\b", cleaned):
        return "sql_agent"
    if re.search(r"\bknowledge\b|\bpolicy\b|\brag\b|\bdocument\b", cleaned):
        return "knowledge_agent"

    # Level 3: If user query mentions document-related words, route to knowledge_agent
    if user_query and DOCUMENT_KEYWORDS.search(user_query):
        logger.info(f"Fallback: user query contains document keywords, routing to knowledge_agent.")
        return "knowledge_agent"

    # Level 4: Fallback — chitchat is the safest default
    logger.warning(f"Could not parse route from LLM response: '{llm_response}'. Defaulting to chitchat.")
    return "chitchat"


def supervisor_node(state: dict) -> dict:
    """
    Supervisor node: classifies the user query and sets the routing decision.
    Respects sql_enabled flag — downgrades sql_agent->chitchat and both->knowledge_agent
    when SQL is disabled.
    Returns dict with 'next_agent' key.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"next_agent": "chitchat"}

    # Get the last user message
    user_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        return {"next_agent": "chitchat"}

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        route = parse_route(response.content, user_query=user_message)

        # Downgrade routes if SQL database is disabled
        sql_enabled = state.get("sql_enabled", True)
        if not sql_enabled:
            if route == "sql_agent":
                logger.info("SQL disabled — downgrading sql_agent to chitchat")
                route = "chitchat"
            elif route == "both":
                logger.info("SQL disabled — downgrading both to knowledge_agent")
                route = "knowledge_agent"

        logger.info(f"Query: '{user_message[:80]}...' -> Route: {route}")
        return {"next_agent": route}

    except Exception as e:
        logger.error(f"Supervisor routing error: {e}")
        return {"next_agent": "chitchat"}
