"""
Thin wrapper around an OpenAI-compatible chat endpoint (LM Studio by default).
Kept as a single small class so it's trivial to point at a different backend
(Ollama, a hosted API, a second "judge" model, etc) later without touching the
agents.
"""
from openai import OpenAI


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, temperature: float = 0.2,
                 max_tokens: int = 200) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class FakeLLMClient:
    """
    Drop-in replacement for LLMClient with no network/model dependency — used by
    the smoke test and available for anyone who wants to unit-test pipeline logic
    without LM Studio running. `responses` is consumed in order, one per call.
    """
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system: str, user: str, temperature: float = 0.2,
                 max_tokens: int = 200) -> str:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            return ""
        return self._responses.pop(0)
