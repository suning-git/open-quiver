"""LLM provider interface and implementations."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the assistant's text response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            The assistant's response text.
        """
        ...


class MockProvider(LLMProvider):
    """Returns pre-scripted responses. For testing only."""

    def __init__(self, responses: list[str]):
        """
        Args:
            responses: List of strings to return in order.
                       Cycles if more calls are made than responses.
        """
        self._responses = responses
        self._call_count = 0

    def chat(self, messages: list[dict]) -> str:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp

    @property
    def call_count(self) -> int:
        return self._call_count

    @classmethod
    def from_actions(cls, actions: list[int]) -> "MockProvider":
        """Convenience: create a MockProvider from a list of vertex numbers."""
        return cls([str(a) for a in actions])


class OpenAICompatProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs (GPT, DeepSeek, Qwen, Claude)."""

    def __init__(self, model: str, base_url: str, api_key: str, **kwargs):
        """
        Args:
            model: Model name (e.g. "deepseek-chat", "gpt-4o").
            base_url: API base URL.
            api_key: API key.
            **kwargs: Extra params passed to client.chat.completions.create().
        """
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._kwargs = kwargs

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._kwargs,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
