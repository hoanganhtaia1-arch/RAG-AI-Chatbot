"""
web_search.py

Web search using Tavily API.

Requires: TAVILY_API_KEY environment variable.

Usage:
    from web_search import search_tavily
    results = search_tavily("học bổng Chevening", max_results=5)
    # returns: [{"url": str, "content": str, "title": str, "score": float}, ...]
"""

from __future__ import annotations
import os
from urllib.parse import urlparse


def _normalize_url(url: str) -> str | None:
    """Normalize and validate a URL. Returns None if invalid."""
    url = (url or "").strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    # Drop fragments; keep query.
    clean = parsed._replace(fragment="").geturl()
    return clean


def search_tavily(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily API.

    Args:
        query       : search query string
        max_results : maximum number of results to return

    Returns:
        list of dicts with keys:
            - url     : str  — result URL
            - content : str  — content snippet from Tavily
            - title   : str  — page title
            - score   : float — relevance score (higher = more relevant)

    Raises:
        ValueError  : if TAVILY_API_KEY is not set
        ImportError : if tavily-python is not installed
    """
    q = (query or "").strip()
    if not q:
        return []

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not set. "
            "Add it to your .env file: TAVILY_API_KEY=tvly-..."
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "tavily-python is not installed. Run: uv add tavily-python"
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=q,
        search_depth="advanced",
        max_results=max_results,
    )

    results: list[dict] = []
    for r in response.get("results", []):
        url = _normalize_url(r.get("url", ""))
        if not url:
            continue
        results.append({
            "url":     url,
            "content": r.get("content", "") or "",
            "title":   r.get("title",   "") or "",
            "score":   float(r.get("score", 0.0)),
        })

    print(f"[WebSearch] Tavily: {len(results)} results for '{q[:60]}'")
    return results
