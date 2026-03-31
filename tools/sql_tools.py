"""
SQL tools for querying the customer support PostgreSQL database.
READ-ONLY enforcement via keyword blocklist + LLM prompt instructions.
"""

import re

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage

from config import POSTGRES_URL, get_llm
from utils import setup_logging

logger = setup_logging("sql_tools")

# Forbidden SQL keywords — defense layer 1
FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
]

FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE
)

SQL_SYSTEM_PROMPT = """You are a read-only SQL assistant for a customer support database. You MUST only generate SELECT queries. NEVER generate DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, or any DDL/DML statements.

The database has three tables:

1. customers (customer_id, first_name, last_name, email, phone, plan, signup_date, status)
   - plan: 'free', 'basic', 'premium', 'enterprise'
   - status: 'active', 'inactive', 'suspended'

2. support_tickets (ticket_id, customer_id, subject, description, status, priority, category, created_at, resolved_at)
   - status: 'open', 'in_progress', 'resolved', 'closed'
   - priority: 'low', 'medium', 'high', 'critical'

3. interactions (interaction_id, ticket_id, agent_name, message, sender, created_at)
   - sender: 'customer', 'agent', 'system'

IMPORTANT: When writing JOIN queries, ALWAYS prefix column names with their table name to avoid ambiguity (e.g., customers.status, support_tickets.status). Never use bare column names in JOIN queries.

When the user asks a question, respond with EXACTLY one SQL SELECT query that answers it. Output ONLY the SQL query, nothing else. Do not wrap in markdown code blocks."""

SQL_RETRY_PROMPT = """The following SQL query failed with this error:

Query: {sql_query}
Error: {error}

Please fix the query and try again. Output ONLY the corrected SQL query, nothing else. Do not wrap in markdown code blocks."""

MAX_SQL_RETRIES = 2

def validate_sql(query: str) -> tuple[bool, str]:
    """Check SQL query for forbidden keywords. Returns (is_valid, error_message)."""
    match = FORBIDDEN_PATTERN.search(query)
    if match:
        return False, f"Blocked: query contains forbidden keyword '{match.group()}'. Only SELECT queries are allowed."
    return True, ""


def get_database() -> SQLDatabase:
    """Create a SQLDatabase instance connected to PostgreSQL."""
    return SQLDatabase.from_uri(POSTGRES_URL)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM-generated SQL."""
    sql = re.sub(r"^```(?:sql)?\s*", "", raw.strip())
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def query_database(natural_language_query: str) -> str:
    """
    Convert a natural language question to SQL, execute it, and return raw results.

    Uses the LLM to generate the SQL query with automatic retry (up to MAX_SQL_RETRIES)
    on execution errors — the LLM sees the Postgres error and self-corrects.
    """
    try:
        db = get_database()
        llm = get_llm()

        # Step 1: Generate SQL from natural language
        messages = [
            SystemMessage(content=SQL_SYSTEM_PROMPT),
            HumanMessage(content=natural_language_query),
        ]
        response = llm.invoke(messages)
        sql_query = _clean_sql(response.content)

        logger.info(f"Generated SQL: {sql_query}")

        # Step 2: Validate — defense layer 1
        is_valid, error_msg = validate_sql(sql_query)
        if not is_valid:
            return error_msg

        # Step 3: Execute with retry loop
        last_error = None
        for attempt in range(1 + MAX_SQL_RETRIES):
            try:
                results = db.run(sql_query)
                return results
            except Exception as e:
                last_error = e
                logger.warning(f"SQL execution error (attempt {attempt + 1}): {e}")

                if attempt < MAX_SQL_RETRIES:
                    # Ask the LLM to fix the query using the error message
                    retry_messages = [
                        SystemMessage(content=SQL_SYSTEM_PROMPT),
                        HumanMessage(content=natural_language_query),
                        # Include the failed attempt so the LLM has context
                        HumanMessage(content=SQL_RETRY_PROMPT.format(
                            sql_query=sql_query,
                            error=str(e),
                        )),
                    ]
                    retry_response = llm.invoke(retry_messages)
                    sql_query = _clean_sql(retry_response.content)

                    logger.info(f"Retry SQL (attempt {attempt + 2}): {sql_query}")

                    # Validate the retried query too
                    is_valid, error_msg = validate_sql(sql_query)
                    if not is_valid:
                        return error_msg

        return f"Could not execute the database query after {1 + MAX_SQL_RETRIES} attempts. Last error: {last_error}"

    except Exception as e:
        logger.error(f"query_database error: {e}")
        return f"I encountered an error while querying the database: {str(e)}. Please ensure PostgreSQL is running."
