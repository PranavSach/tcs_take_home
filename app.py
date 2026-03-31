"""
Streamlit UI for TechCorp Solutions AI Customer Support.
Main entry point: streamlit run app.py
"""

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*__path__.*")

import tempfile
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="TechCorp AI Support",
    page_icon="\U0001f3e2",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-sql { background-color: #dbeafe; color: #1e40af; }
    .badge-kb { background-color: #dcfce7; color: #166534; }
    .badge-both { background-color: #f3e8ff; color: #7c3aed; }
    .badge-chitchat { background-color: #f3f4f6; color: #4b5563; }
    .status-ok { color: #16a34a; }
    .status-err { color: #dc2626; }
    div[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_vector_store():
    from tools.kb_tools import initialize_vector_store
    return initialize_vector_store()


@st.cache_resource
def load_graph():
    from agents.graph import create_graph
    return create_graph()


@st.cache_resource
def load_embeddings():
    from config import get_embeddings
    return get_embeddings()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_routes" not in st.session_state:
    st.session_state.agent_routes = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# Load resources
try:
    vector_store = load_vector_store()
    vs_status = True
except Exception as e:
    vs_status = False
    vs_error = str(e)

try:
    graph = load_graph()
    graph_status = True
except Exception as e:
    graph_status = False
    graph_error = str(e)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_db_connection() -> bool:
    try:
        from config import POSTGRES_URL
        import psycopg2
        conn = psycopg2.connect(POSTGRES_URL)
        conn.close()
        return True
    except Exception:
        return False


def check_llm_connection() -> tuple[bool, str]:
    from config import USE_OPENAI, OLLAMA_MODEL
    if USE_OPENAI:
        return True, "OpenAI GPT-4o"
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            return True, f"Ollama ({OLLAMA_MODEL})"
    except Exception:
        pass
    return False, "Ollama (not connected)"


def get_badge_html(route: str) -> str:
    badge_map = {
        "sql_agent": ('\U0001f5c3\ufe0f SQL Agent', 'badge-sql'),
        "knowledge_agent": ('\U0001f4c4 Knowledge Base', 'badge-kb'),
        "both": ('\U0001f5c3\ufe0f\U0001f4c4 Both', 'badge-both'),
        "chitchat": ('\U0001f4ac General', 'badge-chitchat'),
    }
    label, css_class = badge_map.get(route, ('\U0001f4ac General', 'badge-chitchat'))
    return f'<span class="agent-badge {css_class}">{label}</span>'


def run_query(user_input: str):
    """Process a user query: graph for routing + data, then stream the final LLM response."""
    from config import get_llm

    # Append user message to display history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # MemorySaver handles state persistence across turns via thread_id
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    route_labels = {
        "sql_agent": "\U0001f5c3\ufe0f Querying database...",
        "knowledge_agent": "\U0001f4c4 Searching documents...",
        "both": "\U0001f5c3\ufe0f\U0001f4c4 Gathering data and documents...",
        "chitchat": "\U0001f4ac Composing response...",
    }

    # Build exclude_sources list based on sidebar toggle
    SAMPLE_DOCS = ["company_refund_policy.pdf", "terms_of_service.pdf"]
    use_sample = st.session_state.get("toggle_sample_docs", True)
    sql_enabled = st.session_state.get("toggle_sql_data", True)
    exclude_sources = SAMPLE_DOCS if not use_sample else []

    # Edge case: all data sources disabled and no uploaded documents
    has_uploaded_docs = "last_uploaded_doc" in st.session_state
    if not sql_enabled and not use_sample and not has_uploaded_docs:
        with st.chat_message("assistant"):
            no_sources_msg = (
                "No data sources are currently enabled. "
                "Please enable at least one data source in the sidebar "
                "or upload a document to get started."
            )
            st.warning(no_sources_msg)
        st.session_state.messages.append({"role": "assistant", "content": no_sources_msg})
        st.session_state.agent_routes.append("chitchat")
        return

    try:
        # Phase 1: Run graph (routing + data gathering) — show status
        with st.status("Processing your query...", expanded=False) as status:
            status.update(label="Routing query...")
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "next_agent": "",
                    "final_response": "",
                    "agent_outputs": {},
                    "last_uploaded_doc": st.session_state.get("last_uploaded_doc", ""),
                    "final_prompt": [],
                    "sql_enabled": sql_enabled,
                    "exclude_sources": exclude_sources,
                },
                config=config,
            )

            route = result.get("next_agent", "chitchat")
            status.update(label=route_labels.get(route, "Processing..."))
            status.update(label="Generating response...", state="complete")

        # Phase 2: Stream the final LLM response token by token
        with st.chat_message("assistant"):
            final_prompt = result.get("final_prompt", [])

            if final_prompt:
                streaming_llm = get_llm()
                response = st.write_stream(
                    chunk.content
                    for chunk in streaming_llm.stream(final_prompt)
                    if hasattr(chunk, "content") and chunk.content
                )
            else:
                # Fallback if final_prompt wasn't set
                response = result.get("final_response", "I couldn't process that query.")
                st.markdown(response)

            st.markdown(get_badge_html(route), unsafe_allow_html=True)

        # Save to display history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.agent_routes.append(route)

    except Exception as e:
        error_msg = str(e)
        with st.chat_message("assistant"):
            if "Connection" in error_msg or "refused" in error_msg:
                st.error(
                    "\u26a0\ufe0f Could not connect to the language model. "
                    "Please ensure Ollama is running (`ollama serve`) or set OPENAI_API_KEY in .env"
                )
            else:
                st.error(f"\u26a0\ufe0f An error occurred: {error_msg}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
        st.session_state.agent_routes.append("chitchat")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("\U0001f3e2 TechCorp AI")
    st.caption("Customer Support System")
    st.divider()

    # Document Upload
    st.subheader("\U0001f4c4 Upload Document")
    uploaded_file = st.file_uploader("Upload a policy PDF", type=["pdf"], key="pdf_upload")
    replace_existing = st.checkbox("Replace if already exists", key="replace_doc")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            from tools.kb_tools import upload_document
            result = upload_document(tmp_path, uploaded_file.name, replace=replace_existing)
            if isinstance(result, str):
                # Document was skipped (duplicate)
                st.info(result)
            else:
                st.session_state.last_uploaded_doc = uploaded_file.name
                st.success(f"Uploaded **{uploaded_file.name}** ({result} chunks indexed)")
                # Clear cached vector store so new docs are searchable
                load_vector_store.clear()
                load_vector_store()
        except Exception as e:
            st.error(f"Upload failed: {e}")
        finally:
            os.unlink(tmp_path)

    st.divider()

    # Data Sources
    st.subheader("\U0001f4ca Data Sources")
    use_sample_docs = st.toggle(
        "Sample Policy Docs (Refund Policy, Terms of Service)",
        value=True,
        key="toggle_sample_docs",
    )
    use_sql_data = st.toggle(
        "Customer SQL Database",
        value=True,
        key="toggle_sql_data",
    )

    st.divider()

    # System Status
    st.subheader("\U0001f4ca System Status")
    db_ok = check_db_connection()
    llm_ok, llm_name = check_llm_connection()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**DB:** {'<span class=\"status-ok\">\u2705</span>' if db_ok else '<span class=\"status-err\">\u274c</span>'}", unsafe_allow_html=True)
        st.markdown(f"**VectorDB:** {'<span class=\"status-ok\">\u2705</span>' if vs_status else '<span class=\"status-err\">\u274c</span>'}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**LLM:** {'<span class=\"status-ok\">\U0001f7e2</span>' if llm_ok else '<span class=\"status-err\">\U0001f534</span>'}", unsafe_allow_html=True)
        if llm_ok:
            st.caption(llm_name)

    st.divider()

    # Sample Queries
    st.subheader("\U0001f4cb Sample Queries")
    sample_queries = [
        "What is the current refund policy?",
        "Give me Ema's profile and support ticket details",
        "How many open tickets are there?",
        "What are the terms for account termination?",
        "Which customers are on the enterprise plan?",
    ]

    for query in sample_queries:
        if st.button(query, key=f"sample_{hash(query)}", use_container_width=True):
            st.session_state.pending_query = query
            st.rerun()

    st.divider()
    if st.button("\U0001f5d1\ufe0f Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_routes = []
        st.session_state.thread_id = str(uuid4())  # New thread for fresh memory
        st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.title("\U0001f3e2 TechCorp Solutions \u2014 AI Customer Support")
st.caption("Ask questions about customer data, company policies, or anything else.")

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.agent_routes):
            route_idx = i // 2
            if route_idx < len(st.session_state.agent_routes):
                st.markdown(
                    get_badge_html(st.session_state.agent_routes[route_idx]),
                    unsafe_allow_html=True
                )

# Handle pending query from sample buttons
if "pending_query" in st.session_state:
    pending = st.session_state.pending_query
    del st.session_state.pending_query
    run_query(pending)

# Chat input
user_input = st.chat_input("Ask a question...")
if user_input:
    run_query(user_input)
