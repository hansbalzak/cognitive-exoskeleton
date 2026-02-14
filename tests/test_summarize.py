import pytest
from ai_agent import SimpleAI

def test_summarize_conversation(mocker):
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.conversation = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message 1"},
        {"role": "user", "content": "User message 2"},
        {"role": "assistant", "content": "Assistant message 2"},
    ]

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Summary of the conversation."
                }
            }
        ]
    }
    mocker.patch.object(ai, '_post_chat', return_value=(True, mock_response))

    summary = ai.summarize_conversation()
    assert summary == "Summary of the conversation."
