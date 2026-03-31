import logging
import sys


def setup_logging(name: str = "customer_support_ai", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_sources(sources: list[dict]) -> str:
    """Format source citations from knowledge base results."""
    if not sources:
        return ""
    citations = []
    for src in sources:
        source_name = src.get("source", "Unknown")
        page = src.get("page", "N/A")
        citations.append(f"- {source_name}, page {page}")
    return "\n**Sources:**\n" + "\n".join(citations)
