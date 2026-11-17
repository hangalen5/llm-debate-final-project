import os
import pytest
from src.chat import call_deepseek

@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set in environment variables"
)
def test_call_deepseek_response():
    """
    Verify that call_deepseek returns a valid, non-empty string response.
    """
    prompt = "Reply with the exact word: READY"
    response = call_deepseek(prompt, temperature=0.0)

    assert isinstance(response, str)
    assert len(response.strip()) > 0