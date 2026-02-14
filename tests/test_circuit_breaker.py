import pytest
from ai_agent import SimpleAI

def test_circuit_breaker():
    ai = SimpleAI(base_url="", model="", temperature=0.0, max_tokens=100)
    ai.consecutive_failures = 2
    ai.circuit_breaker_active = False
    ai.circuit_breaker_end_time = 0

    ai._update_circuit_breaker_state()
    assert ai.circuit_breaker_active == False

    ai.consecutive_failures = 3
    ai._update_circuit_breaker_state()
    assert ai.circuit_breaker_active == True
    assert ai.circuit_breaker_end_time > time.time()

    time.sleep(1)
    ai._update_circuit_breaker_state()
    assert ai.circuit_breaker_active == False
    assert ai.consecutive_failures == 0
