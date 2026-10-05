"""Unit tests for FastAPI server and data collection endpoints."""
import pytest
from fastapi.testclient import TestClient

from airscript.server.app import app

client = TestClient(app)


def test_root_and_collect_pages():
    resp = client.get("/collect")
    assert resp.status_code == 200
    assert "AirScript" in resp.text


def test_collect_sample_and_stats():
    # Valid sample payload
    payload = {
        "label": "B",
        "user_id": "test_user",
        "strokes": [
            [
                {"x": 0.2, "y": 0.1, "z": 0.0, "t": 0.0, "p": 1},
                {"x": 0.2, "y": 0.5, "z": 0.0, "t": 0.1, "p": 1},
                {"x": 0.4, "y": 0.3, "z": 0.0, "t": 0.2, "p": 1},
                {"x": 0.2, "y": 0.5, "z": 0.0, "t": 0.3, "p": 1},
            ]
        ],
        "metadata": {"test": True},
    }

    save_resp = client.post("/api/collect/save", json=payload)
    assert save_resp.status_code == 200
    data = save_resp.json()
    assert data["status"] == "success"
    assert data["points"] == 4

    stats_resp = client.get("/api/collect/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_samples"] >= 1
    assert "B" in stats["samples_by_class"]


def test_websocket_recognize():
    with client.websocket_connect("/ws/recognize") as ws:
        # Send empty landmarks frame
        ws.send_json({"type": "landmarks", "landmarks": [], "timestamp": 0.0})
        data = ws.receive_json()
        assert data["type"] == "frame_update"
        assert data["hand_detected"] is False

        # Send clear command
        ws.send_json({"type": "clear"})
        clear_data = ws.receive_json()
        assert clear_data["type"] == "cleared"
