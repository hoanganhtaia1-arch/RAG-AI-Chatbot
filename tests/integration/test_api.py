import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check_integration():
    """Test health check and Qdrant DB connection status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "qdrant_ok" in data
    assert "flags" in data

def test_cache_stats_integration():
    """Test cache stats endpoint."""
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    
def test_cache_clear_integration():
    """Test cache clear endpoint."""
    response = client.post("/cache/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cleared"

def test_ingest_url_missing_body():
    """Test ingest URL API with missing payload."""
    response = client.post("/ingest/url", json={})
    assert response.status_code == 400
    assert "Missing 'url'" in response.json()["detail"]
