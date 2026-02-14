import pytest
from ai_agent import _safe_load_json, _atomic_write_json, _atomic_write_text, SimpleAI

def test_load_save_conversation(tmp_path):
    conversation_path = tmp_path / "conversation.json"
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.conversation_path = conversation_path
    ai.conversation = [{"role": "user", "content": "Hello"}]
    ai.save_conversation()
    loaded_conversation = ai.load_conversation()
    assert loaded_conversation == ai.conversation

def test_load_save_profile(tmp_path):
    profile_path = tmp_path / "profile.json"
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.profile_path = profile_path
    ai.save_profile({"name": "Test"})
    loaded_profile = ai.load_profile()
    assert loaded_profile == {"name": "Test"}

def test_load_save_identity(tmp_path):
    identity_path = tmp_path / "identity.json"
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.identity_path = identity_path
    ai.save_identity({"name": "Test"})
    loaded_identity = ai.load_identity()
    assert loaded_identity == {"name": "Test"}

def test_load_save_facts(tmp_path):
    facts_path = tmp_path / "facts.jsonl"
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.facts_path = facts_path
    ai.save_facts([{"id": 1, "fact": "Test fact"}])
    loaded_facts = ai.load_facts()
    assert loaded_facts == [{"id": 1, "fact": "Test fact"}]

def test_corrupt_recovery(tmp_path):
    conversation_path = tmp_path / "conversation.json"
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.conversation_path = conversation_path
    ai.conversation = [{"role": "user", "content": "Hello"}]
    ai.save_conversation()
    conversation_path.write_text("Invalid JSON", encoding="utf-8")
    loaded_conversation = ai.load_conversation()
    assert loaded_conversation == []
