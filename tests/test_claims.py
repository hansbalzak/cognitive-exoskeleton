import pytest
from ai_agent import SimpleAI

def test_append_claims(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.claims_path = claims_path
    ai.append_claims([{"claim": "Test claim", "topic": "Test topic"}])
    loaded_claims = ai.load_claims()
    assert len(loaded_claims) == 1
    assert loaded_claims[0]["claim"] == "Test claim"
    assert loaded_claims[0]["topic"] == "Test topic"
    assert loaded_claims[0]["id"] == 1
    assert loaded_claims[0]["confidence"] == 0.6
    assert loaded_claims[0]["status"] == "active"
    assert "created_at" in loaded_claims[0]
    assert "last_seen_at" in loaded_claims[0]
    assert loaded_claims[0]["source"] == "assistant"
    assert loaded_claims[0]["corrections"] == []

    ai.append_claims([{"claim": "Another claim", "topic": "Another topic"}])
    loaded_claims = ai.load_claims()
    assert len(loaded_claims) == 2
    assert loaded_claims[1]["id"] == 2
