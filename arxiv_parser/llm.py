"""LLM client: minimal chat-completion provider abstraction.

Only the standard library is used. The OpenAI-compatible provider works
with any /chat/completions endpoint (OpenAI, Groq, DeepSeek, Ollama,
LM Studio, ...). New providers can be added as small subclasses.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .console import say


class LLMError(RuntimeError):
    """Raised when the LLM API call fails or returns an unexpected shape."""


class LLMProvider:
    """Minimal chat-completion interface."""

    name: str = "abstract"

    def complete(self, system: str, user: str, temperature: float) -> str:
        raise NotImplementedError


class FallbackProvider(LLMProvider):
    """Tries a chain of providers in order, switching when one fails.

    Useful for free-tier models that return transient 'high demand'
    errors: the request falls through to a quieter model instead of
    failing the run.
    """

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers

    @property
    def name(self) -> str:  # type: ignore[override]
        return " -> ".join(p.name for p in self.providers)

    def complete(self, system: str, user: str, temperature: float) -> str:
        errors: list[str] = []
        for i, provider in enumerate(self.providers):
            if i > 0:
                prev = str(errors[-1]).splitlines()[0][:120]
                say(f"  switching to fallback model {provider.name} ({prev})")
            try:
                return provider.complete(system, user, temperature)
            except LLMError as e:
                errors.append(f"{provider.name}: {e}")
        raise LLMError(" | ".join(errors))


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible POST /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"openai-compatible ({self.model})"

    def complete(self, system: str, user: str, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            detail = raw
            try:
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    detail = data.get("error", {}).get("message") or raw
            except (json.JSONDecodeError, AttributeError):
                pass
            raise LLMError(
                f"LLM API returned HTTP {e.code}: {detail.strip()[:300]}"
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"LLM API unreachable: {e}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                f"Unexpected LLM API response: {json.dumps(data)[:300]}"
            ) from e
