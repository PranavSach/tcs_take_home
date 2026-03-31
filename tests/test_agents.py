"""
Smoke tests for the customer support AI multi-agent system.
Tests routing logic, SQL validation, and knowledge base search.

Run: python -m pytest tests/test_agents.py -v
"""

import unittest

from agents.supervisor import parse_route
from tools.sql_tools import validate_sql


class TestSupervisorRouting(unittest.TestCase):
    """Test the defensive parsing cascade for supervisor routing."""

    def test_exact_match_sql_agent(self):
        self.assertEqual(parse_route("sql_agent"), "sql_agent")

    def test_exact_match_knowledge_agent(self):
        self.assertEqual(parse_route("knowledge_agent"), "knowledge_agent")

    def test_exact_match_both(self):
        self.assertEqual(parse_route("both"), "both")

    def test_exact_match_chitchat(self):
        self.assertEqual(parse_route("chitchat"), "chitchat")

    def test_exact_match_with_whitespace(self):
        self.assertEqual(parse_route("  sql_agent  "), "sql_agent")
        self.assertEqual(parse_route("\nknowledge_agent\n"), "knowledge_agent")

    def test_exact_match_with_quotes(self):
        self.assertEqual(parse_route('"sql_agent"'), "sql_agent")
        self.assertEqual(parse_route("'both'"), "both")

    def test_keyword_search_in_sentence(self):
        self.assertEqual(parse_route("I think this should go to sql_agent"), "sql_agent")
        self.assertEqual(parse_route("Route to knowledge_agent for policy info"), "knowledge_agent")

    def test_keyword_search_both_priority(self):
        # "both" should be checked before individual agents
        self.assertEqual(parse_route("This needs both agents"), "both")

    def test_partial_keyword_sql(self):
        self.assertEqual(parse_route("This is a SQL query about customers"), "sql_agent")

    def test_partial_keyword_knowledge(self):
        self.assertEqual(parse_route("This is about company policy"), "knowledge_agent")
        self.assertEqual(parse_route("Search the document for this"), "knowledge_agent")

    def test_fallback_to_chitchat(self):
        self.assertEqual(parse_route("I have no idea what this is"), "chitchat")
        self.assertEqual(parse_route(""), "chitchat")

    def test_document_keyword_fallback(self):
        # When LLM response is unparseable but user query mentions document keywords,
        # route to knowledge_agent instead of chitchat
        self.assertEqual(parse_route("unclear response", user_query="Tell me about this pdf"), "knowledge_agent")
        self.assertEqual(parse_route("hmm", user_query="What's in the uploaded document?"), "knowledge_agent")
        self.assertEqual(parse_route("hmm", user_query="Summarize the file"), "knowledge_agent")
        self.assertEqual(parse_route("hmm", user_query="What is the policy on refunds?"), "knowledge_agent")
        # Without document keywords, still falls back to chitchat
        self.assertEqual(parse_route("unclear response", user_query="hello there"), "chitchat")

    def test_case_insensitive(self):
        self.assertEqual(parse_route("SQL_AGENT"), "sql_agent")
        self.assertEqual(parse_route("KNOWLEDGE_AGENT"), "knowledge_agent")
        self.assertEqual(parse_route("BOTH"), "both")


class TestSQLValidation(unittest.TestCase):
    """Test SQL query validation for read-only enforcement."""

    def test_select_allowed(self):
        is_valid, _ = validate_sql("SELECT * FROM customers")
        self.assertTrue(is_valid)

    def test_select_with_join_allowed(self):
        is_valid, _ = validate_sql(
            "SELECT c.first_name, t.subject FROM customers c "
            "JOIN support_tickets t ON c.customer_id = t.customer_id"
        )
        self.assertTrue(is_valid)

    def test_select_with_where_allowed(self):
        is_valid, _ = validate_sql("SELECT * FROM customers WHERE plan = 'premium'")
        self.assertTrue(is_valid)

    def test_count_allowed(self):
        is_valid, _ = validate_sql("SELECT COUNT(*) FROM support_tickets WHERE status = 'open'")
        self.assertTrue(is_valid)

    def test_drop_blocked(self):
        is_valid, msg = validate_sql("DROP TABLE customers")
        self.assertFalse(is_valid)
        self.assertIn("DROP", msg)

    def test_delete_blocked(self):
        is_valid, msg = validate_sql("DELETE FROM customers WHERE customer_id = 1")
        self.assertFalse(is_valid)
        self.assertIn("DELETE", msg)

    def test_update_blocked(self):
        is_valid, msg = validate_sql("UPDATE customers SET plan = 'free' WHERE customer_id = 1")
        self.assertFalse(is_valid)
        self.assertIn("UPDATE", msg)

    def test_insert_blocked(self):
        is_valid, msg = validate_sql("INSERT INTO customers (first_name) VALUES ('Hack')")
        self.assertFalse(is_valid)
        self.assertIn("INSERT", msg)

    def test_alter_blocked(self):
        is_valid, msg = validate_sql("ALTER TABLE customers ADD COLUMN hacked BOOLEAN")
        self.assertFalse(is_valid)
        self.assertIn("ALTER", msg)

    def test_truncate_blocked(self):
        is_valid, msg = validate_sql("TRUNCATE TABLE customers")
        self.assertFalse(is_valid)
        self.assertIn("TRUNCATE", msg)

    def test_case_insensitive_blocking(self):
        is_valid, _ = validate_sql("drop table customers")
        self.assertFalse(is_valid)

        is_valid, _ = validate_sql("Delete FROM customers")
        self.assertFalse(is_valid)


class TestGraphRouting(unittest.TestCase):
    """Test the graph routing functions."""

    def test_route_from_supervisor_sql(self):
        from agents.graph import route_from_supervisor
        state = {"next_agent": "sql_agent"}
        self.assertEqual(route_from_supervisor(state), "sql_agent")

    def test_route_from_supervisor_knowledge(self):
        from agents.graph import route_from_supervisor
        state = {"next_agent": "knowledge_agent"}
        self.assertEqual(route_from_supervisor(state), "knowledge_agent")

    def test_route_from_supervisor_both_maps_to_sql(self):
        from agents.graph import route_from_supervisor
        state = {"next_agent": "both"}
        self.assertEqual(route_from_supervisor(state), "sql_agent")

    def test_route_from_supervisor_chitchat(self):
        from agents.graph import route_from_supervisor
        state = {"next_agent": "chitchat"}
        self.assertEqual(route_from_supervisor(state), "chitchat")

    def test_route_after_sql_both_chains(self):
        from agents.graph import route_after_sql
        state = {"next_agent": "both"}
        self.assertEqual(route_after_sql(state), "knowledge_agent")

    def test_route_after_sql_single_goes_to_synthesizer(self):
        from agents.graph import route_after_sql
        state = {"next_agent": "sql_agent"}
        self.assertEqual(route_after_sql(state), "synthesizer")


if __name__ == "__main__":
    unittest.main()
