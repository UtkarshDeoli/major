# Track A: LLM Provider Abstraction + Structured Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gemini-only AI layer with a provider-agnostic LLM abstraction (OpenRouter + modern Gemini SDK), add validated structured outputs with retry/fallback, and migrate chat, analysis, and mock-test generation to use it.

**Architecture:** A small `src/services/llm/` package exposes a protocol, two provider implementations, a structured-output parser, and a high-level `LLMService`. Existing API contracts stay unchanged; call sites are migrated behind the new service.

**Tech Stack:** FastAPI, Pydantic, OpenAI SDK (`openai>=1.12.0`), Google Gen AI SDK (`google-genai>=2.8.0`), pytest.

---

## File Structure

### New files

- `src/services/llm/__init__.py`
- `src/services/llm/protocol.py`
- `src/services/llm/config.py`
- `src/services/llm/structured_output.py`
- `src/services/llm/openrouter_provider.py`
- `src/services/llm/gemini_provider.py`
- `src/services/llm/llm_service.py`
- `tests/services/llm/test_structured_output.py`
- `tests/services/llm/test_openrouter_provider.py`
- `tests/services/llm/test_gemini_provider.py`
- `tests/services/llm/test_llm_service.py`

### Modified files

- `requirements.txt`
- `src/core/config.py`
- `src/core/models.py`
- `src/services/llm_service.py`
- `src/services/gemini_service.py`
- `src/services/question_paper_analysis_service.py`
- `src/services/mock_test_service.py`

---

## Task 1: Update dependencies

**Files:**
- Modify: `requirements.txt:18-20`

- [ ] **Step 1: Replace legacy Gemini SDK and add OpenAI SDK**

Replace:
```text
chromadb>=0.4.22
sentence-transformers>=2.3.0
google-generativeai>=0.4.0
```
With:
```text
chromadb>=0.4.22
sentence-transformers>=2.3.0
google-genai>=2.8.0
openai>=1.12.0
```

- [ ] **Step 2: Install dependencies in the backend virtual environment**

Run:
```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: installs `google-genai` and `openai` without errors.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add google-genai and openai for provider-agnostic LLM layer"
```

---

## Task 2: Add LLM configuration to core config

**Files:**
- Modify: `src/core/config.py:1-44`

- [ ] **Step 1: Add LLM provider and model env vars to Settings**

Replace the existing `Settings` class with the expanded version below.

```python
import os
from typing import Optional, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "phadai"
    MONGODB_CONNECT_TIMEOUT: int = 30000

    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ChromaDB settings
    CHROMA_DB_PATH: str = "./chroma_db"
    GEMINI_API_KEY: str

    # External search / agent settings
    WEB_SEARCH_API_KEY: Optional[str] = None
    WEB_SEARCH_ENGINE: str = "serper"

    # LLM provider settings
    LLM_PROVIDER: Literal["openrouter", "gemini"] = "openrouter"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    OR_MODEL_CHAT: str = "meta-llama/llama-3.3-70b-instruct"
    OR_MODEL_ANALYSIS: str = "deepseek/deepseek-r1-0528"
    OR_MODEL_QUIZ: str = "stepfun/step-3.5-flash"
    OR_MODEL_FALLBACK: str = "openrouter/free"

    GEMINI_MODEL_CHAT: str = "gemini-2.5-flash"
    GEMINI_MODEL_ANALYSIS: str = "gemini-2.5-flash"
    GEMINI_MODEL_QUIZ: str = "gemini-2.5-flash"
    GEMINI_MODEL_FALLBACK: str = "gemini-2.5-flash"

    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 8192
    LLM_TIMEOUT_SECONDS: float = 120.0
    LLM_MAX_RETRIES: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

settings = Settings()
```

- [ ] **Step 2: Update exported module variables**

Append the following to the bottom of `src/core/config.py`:

```python
# Export LLM settings
LLM_PROVIDER = settings.LLM_PROVIDER
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = settings.OPENROUTER_BASE_URL
OR_MODEL_CHAT = settings.OR_MODEL_CHAT
OR_MODEL_ANALYSIS = settings.OR_MODEL_ANALYSIS
OR_MODEL_QUIZ = settings.OR_MODEL_QUIZ
OR_MODEL_FALLBACK = settings.OR_MODEL_FALLBACK
GEMINI_MODEL_CHAT = settings.GEMINI_MODEL_CHAT
GEMINI_MODEL_ANALYSIS = settings.GEMINI_MODEL_ANALYSIS
GEMINI_MODEL_QUIZ = settings.GEMINI_MODEL_QUIZ
GEMINI_MODEL_FALLBACK = settings.GEMINI_MODEL_FALLBACK
LLM_TEMPERATURE = settings.LLM_TEMPERATURE
LLM_MAX_TOKENS = settings.LLM_MAX_TOKENS
LLM_TIMEOUT_SECONDS = settings.LLM_TIMEOUT_SECONDS
LLM_MAX_RETRIES = settings.LLM_MAX_RETRIES
```

- [ ] **Step 3: Add example env vars to `.env.example`**

Append to `Backend/.env.example`:

```text
# LLM Provider (openrouter | gemini)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional model overrides
# OR_MODEL_CHAT=meta-llama/llama-3.3-70b-instruct
# OR_MODEL_ANALYSIS=deepseek/deepseek-r1-0528
# OR_MODEL_QUIZ=stepfun/step-3.5-flash
# OR_MODEL_FALLBACK=openrouter/free

# GEMINI_MODEL_CHAT=gemini-2.5-flash
# GEMINI_MODEL_ANALYSIS=gemini-2.5-flash
# GEMINI_MODEL_QUIZ=gemini-2.5-flash
# GEMINI_MODEL_FALLBACK=gemini-2.5-flash

LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=8192
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=3
```

- [ ] **Step 4: Commit**

```bash
git add src/core/config.py Backend/.env.example
git commit -m "config: add LLM provider and model configuration"
```

---

## Task 3: Define the LLM protocol

**Files:**
- Create: `src/services/llm/protocol.py`
- Create: `src/services/llm/__init__.py`

- [ ] **Step 1: Create protocol module**

```python
from dataclasses import dataclass
from typing import Protocol, AsyncIterator

from pydantic import BaseModel


@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMCompletion:
    content: str


@dataclass
class LLMStreamChunk:
    content: str
    is_finished: bool = False


class LLMClient(Protocol):
    """Async, provider-agnostic LLM client protocol."""

    async def complete(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletion:
        ...

    async def stream(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        ...

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> BaseModel:
        ...
```

- [ ] **Step 2: Create package init**

```python
from src.services.llm.protocol import LLMClient, LLMCompletion, LLMMessage, LLMStreamChunk

__all__ = [
    "LLMClient",
    "LLMCompletion",
    "LLMMessage",
    "LLMStreamChunk",
]
```

- [ ] **Step 3: Commit**

```bash
git add src/services/llm/protocol.py src/services/llm/__init__.py
git commit -m "feat(llm): define provider-agnostic LLM protocol"
```

---

## Task 4: Build the structured output parser

**Files:**
- Create: `src/services/llm/structured_output.py`
- Test: `tests/services/llm/test_structured_output.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pydantic import BaseModel
from src.services.llm.structured_output import StructuredOutputParser, parse_json_text

class SampleOutput(BaseModel):
    name: str
    score: int

    @classmethod
    def fallback(cls):
        return cls(name="fallback", score=0)


def test_parse_raw_json():
    text = '{"name": "alice", "score": 42}'
    result = parse_json_text(text)
    assert result == {"name": "alice", "score": 42}


def test_parse_markdown_wrapped_json():
    text = "```json\n{\"name\": \"alice\", \"score\": 42}\n```"
    result = parse_json_text(text)
    assert result == {"name": "alice", "score": 42}


def test_parse_with_surrounding_text():
    text = "Here is the result:\n```json\n{\"name\": \"alice\", \"score\": 42}\n```\nHope that helps."
    result = parse_json_text(text)
    assert result == {"name": "alice", "score": 42}


def test_parse_invalid_returns_none():
    text = "No JSON here"
    result = parse_json_text(text)
    assert result is None


@pytest.mark.asyncio
async def test_parser_validates_success():
    parser = StructuredOutputParser(max_retries=1)
    output = await parser.parse(
        raw_text='{"name": "alice", "score": 42}',
        response_model=SampleOutput,
        build_prompt=lambda _: "prompt",
        generate=lambda _: '{"name": "alice", "score": 42}',
    )
    assert output.name == "alice"
    assert output.score == 42


@pytest.mark.asyncio
async def test_parser_retries_then_fallbacks():
    attempts = []

    async def generate(prompt: str) -> str:
        attempts.append(prompt)
        return "not valid json"

    parser = StructuredOutputParser(max_retries=2)
    output = await parser.parse(
        raw_text="not valid json",
        response_model=SampleOutput,
        build_prompt=lambda previous: f"retry: {previous}",
        generate=generate,
    )

    assert output == SampleOutput.fallback()
    assert len(attempts) == 2  # initial + 1 retry
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source .venv/bin/activate
pytest tests/services/llm/test_structured_output.py -v
```

Expected: FAIL — module and parser not defined.

- [ ] **Step 3: Implement the parser**

```python
import json
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown JSON fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> str | None:
    """Extract the largest balanced JSON object from text."""
    text = strip_markdown_fences(text)
    if not text:
        return None

    # Fast path: text is already a single JSON object
    start = text.find("{")
    if start == -1:
        return None

    # Find the matching closing brace for the first object
    depth = 0
    in_string = False
    escape = False
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            continue
        if ch == '"' and in_string:
            in_string = False
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None
    return text[start:end]


def parse_json_text(text: str) -> dict | None:
    """Try to parse a JSON object from model output."""
    json_text = extract_json_object(text)
    if json_text is None:
        return None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


class StructuredOutputParser:
    """Parse model output into a Pydantic model with retry and fallback."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def parse(
        self,
        raw_text: str,
        response_model: type[T],
        build_prompt: Callable[[str | None], str],
        generate: Callable[[str], str],
    ) -> T:
        """Validate raw_text; retry if invalid; finally return fallback instance."""
        last_error: Exception | None = None
        previous_attempt = raw_text

        for attempt in range(self.max_retries + 1):
            data = parse_json_text(previous_attempt)
            if data is not None:
                try:
                    return response_model(**data)
                except ValidationError as exc:
                    last_error = exc

            # Retry with a stricter prompt
            if attempt < self.max_retries:
                previous_attempt = await generate(build_prompt(previous_attempt))

        if hasattr(response_model, "fallback") and callable(response_model.fallback):
            return response_model.fallback()

        raise last_error or ValueError("Could not parse structured output and no fallback defined")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/services/llm/test_structured_output.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/llm/structured_output.py tests/services/llm/test_structured_output.py
git commit -m "feat(llm): add structured output parser with retry and fallback"
```

---

## Task 5: Implement OpenRouter provider

**Files:**
- Create: `src/services/llm/openrouter_provider.py`
- Test: `tests/services/llm/test_openrouter_provider.py`

- [ ] **Step 1: Create provider config helper**

In `src/services/llm/config.py`:

```python
from src.core.config import (
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OR_MODEL_CHAT,
    OR_MODEL_ANALYSIS,
    OR_MODEL_QUIZ,
    OR_MODEL_FALLBACK,
    GEMINI_MODEL_CHAT,
    GEMINI_MODEL_ANALYSIS,
    GEMINI_MODEL_QUIZ,
    GEMINI_MODEL_FALLBACK,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
)


def get_model_mapping(provider: str) -> dict[str, str]:
    if provider == "openrouter":
        return {
            "chat": OR_MODEL_CHAT,
            "analysis": OR_MODEL_ANALYSIS,
            "quiz": OR_MODEL_QUIZ,
            "fallback": OR_MODEL_FALLBACK,
        }
    return {
        "chat": GEMINI_MODEL_CHAT,
        "analysis": GEMINI_MODEL_ANALYSIS,
        "quiz": GEMINI_MODEL_QUIZ,
        "fallback": GEMINI_MODEL_FALLBACK,
    }
```

- [ ] **Step 2: Write the failing test**

```python
import pytest
from unittest.mock import AsyncMock, patch

from src.services.llm.openrouter_provider import OpenRouterProvider
from src.services.llm.protocol import LLMMessage


@pytest.mark.asyncio
async def test_complete_returns_content():
    provider = OpenRouterProvider(api_key="test-key")
    fake_response = AsyncMock()
    fake_response.choices = [AsyncMock(message=AsyncMock(content="hello"))]

    with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock, return_value=fake_response):
        result = await provider.complete(messages=[LLMMessage(role="user", content="hi")])
        assert result.content == "hello"


@pytest.mark.asyncio
async def test_stream_yields_chunks():
    provider = OpenRouterProvider(api_key="test-key")

    async def fake_stream():
        class C:
            choices = [type("D", (), {"delta": type("E", (), {"content": "world"})()})]
        yield C()

    with patch.object(provider._client.chat.completions, "create", new_callable=AsyncMock, return_value=fake_stream()):
        chunks = []
        async for chunk in await provider.stream(messages=[LLMMessage(role="user", content="hi")]):
            chunks.append(chunk)
        assert chunks[0].content == "world"
```

- [ ] **Step 3: Implement the provider**

```python
import json
import os
from typing import AsyncIterator

import openai
from pydantic import BaseModel

from src.services.llm.config import get_model_mapping, LLM_MAX_RETRIES
from src.services.llm.protocol import LLMClient, LLMCompletion, LLMMessage, LLMStreamChunk
from src.services.llm.structured_output import StructuredOutputParser


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model_mapping: dict[str, str] | None = None,
        default_temperature: float = 0.3,
        default_max_tokens: int = 8192,
        default_timeout: float = 120.0,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OpenRouter API key is required")
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=key,
            timeout=default_timeout,
        )
        self._model_mapping = model_mapping or get_model_mapping("openrouter")
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._default_timeout = default_timeout

    def _resolve_model(self, model_key: str) -> str:
        return self._model_mapping.get(model_key, self._model_mapping.get("fallback", "openrouter/free"))

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletion:
        response = await self._client.with_options(
            timeout=timeout_seconds or self._default_timeout
        ).chat.completions.create(
            model=self._resolve_model(model_key),
            messages=self._to_openai_messages(messages),
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
        )
        return LLMCompletion(content=response.choices[0].message.content or "")

    async def stream(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        response = await self._client.with_options(
            timeout=timeout_seconds or self._default_timeout
        ).chat.completions.create(
            model=self._resolve_model(model_key),
            messages=self._to_openai_messages(messages),
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
            stream=True,
        )

        async def _generator():
            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                yield LLMStreamChunk(content=delta)
            yield LLMStreamChunk(content="", is_finished=True)

        return _generator()

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        system_prompt = (
            "You are a helpful assistant that responds only with valid JSON.\n"
            "Do not wrap the JSON in markdown fences or add explanatory text.\n"
            f"Respond with a single JSON object matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        full_messages = [LLMMessage(role="system", content=system_prompt)] + messages

        async def _generate(prompt: str) -> str:
            result = await self.complete(
                [LLMMessage(role="user", content=prompt)],
                model_key=model_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            return result.content

        initial_text = await _generate("\n".join(m.content for m in messages))

        def _build_prompt(previous: str | None) -> str:
            base = "\n".join(m.content for m in messages)
            if previous:
                return (
                    f"Your previous response was invalid. Here it was:\n{previous}\n\n"
                    f"Respond again with ONLY raw JSON matching this schema, no markdown:\n"
                    f"{json.dumps(schema, indent=2)}\n\nPrompt:\n{base}"
                )
            return base

        parser = StructuredOutputParser(max_retries=LLM_MAX_RETRIES)
        return await parser.parse(
            raw_text=initial_text,
            response_model=response_model,
            build_prompt=_build_prompt,
            generate=_generate,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/services/llm/test_openrouter_provider.py -v
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/llm/config.py src/services/llm/openrouter_provider.py tests/services/llm/test_openrouter_provider.py
git commit -m "feat(llm): implement OpenRouter provider with structured output"
```

---

## Task 6: Implement Gemini provider

**Files:**
- Create: `src/services/llm/gemini_provider.py`
- Test: `tests/services/llm/test_gemini_provider.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.llm.gemini_provider import GeminiProvider
from src.services.llm.protocol import LLMMessage
from pydantic import BaseModel


class Sample(BaseModel):
    x: int

    @classmethod
    def fallback(cls):
        return cls(x=0)


@pytest.mark.asyncio
async def test_complete_returns_content():
    provider = GeminiProvider(api_key="test-key")
    fake_response = MagicMock()
    fake_response.text = "hello"

    with patch.object(provider._client.models, "generate_content", new_callable=AsyncMock, return_value=fake_response):
        result = await provider.complete(messages=[LLMMessage(role="user", content="hi")])
        assert result.content == "hello"


@pytest.mark.asyncio
async def test_generate_structured_uses_json_schema():
    provider = GeminiProvider(api_key="test-key")
    fake_response = MagicMock()
    fake_response.text = '{"x": 7}'

    with patch.object(provider._client.models, "generate_content", new_callable=AsyncMock, return_value=fake_response):
        result = await provider.generate_structured(
            messages=[LLMMessage(role="user", content="make a sample")],
            response_model=Sample,
            model_key="chat",
        )
        assert result.x == 7
```

- [ ] **Step 2: Implement the provider**

```python
import os
from typing import AsyncIterator

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.services.llm.config import get_model_mapping, LLM_MAX_RETRIES
from src.services.llm.protocol import LLMClient, LLMCompletion, LLMMessage, LLMStreamChunk
from src.services.llm.structured_output import StructuredOutputParser


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model_mapping: dict[str, str] | None = None,
        default_temperature: float = 0.3,
        default_max_tokens: int = 8192,
        default_timeout: float = 120.0,
    ):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API key is required")
        self._client = genai.Client(api_key=key)
        self._model_mapping = model_mapping or get_model_mapping("gemini")
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._default_timeout = default_timeout

    def _resolve_model(self, model_key: str) -> str:
        return self._model_mapping.get(model_key, self._model_mapping.get("fallback", "gemini-2.5-flash"))

    def _to_gemini_contents(self, messages: list[LLMMessage]) -> list:
        contents = []
        for m in messages:
            if m.role == "system":
                # Gemini handles system instruction via config, not contents.
                continue
            role = m.role if m.role in ("user", "model") else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))
        return contents

    def _extract_system_instruction(self, messages: list[LLMMessage]) -> str | None:
        system_parts = [m.content for m in messages if m.role == "system"]
        return "\n".join(system_parts) if system_parts else None

    def _build_config(
        self,
        temperature: float | None,
        max_tokens: int | None,
        response_schema: type[BaseModel] | None = None,
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._default_temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
        )
        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = response_schema.model_json_schema()
        return config

    async def complete(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletion:
        config = self._build_config(temperature, max_tokens)
        system_instruction = self._extract_system_instruction(messages)
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self._client.aio.models.generate_content(
            model=self._resolve_model(model_key),
            contents=self._to_gemini_contents(messages),
            config=config,
        )
        return LLMCompletion(content=response.text or "")

    async def stream(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        config = self._build_config(temperature, max_tokens)
        system_instruction = self._extract_system_instruction(messages)
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self._client.aio.models.generate_content_stream(
            model=self._resolve_model(model_key),
            contents=self._to_gemini_contents(messages),
            config=config,
        )

        async def _generator():
            async for chunk in response:
                yield LLMStreamChunk(content=chunk.text or "")
            yield LLMStreamChunk(content="", is_finished=True)

        return _generator()

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> BaseModel:
        config = self._build_config(temperature, max_tokens, response_schema=response_model)
        system_instruction = self._extract_system_instruction(messages)
        if system_instruction:
            config.system_instruction = system_instruction

        async def _generate(prompt: str) -> str:
            response = await self._client.aio.models.generate_content(
                model=self._resolve_model(model_key),
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config,
            )
            return response.text or ""

        initial_text = await _generate("\n".join(m.content for m in messages if m.role != "system"))

        def _build_prompt(previous: str | None) -> str:
            base = "\n".join(m.content for m in messages if m.role != "system")
            if previous:
                return (
                    f"Your previous response was invalid JSON:\n{previous}\n\n"
                    f"Respond again with ONLY valid JSON matching the required schema.\n\n{base}"
                )
            return base

        parser = StructuredOutputParser(max_retries=LLM_MAX_RETRIES)
        return await parser.parse(
            raw_text=initial_text,
            response_model=response_model,
            build_prompt=_build_prompt,
            generate=_generate,
        )
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/services/llm/test_gemini_provider.py -v
```

Expected: tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/services/llm/gemini_provider.py tests/services/llm/test_gemini_provider.py
git commit -m "feat(llm): implement Gemini provider with google-genai and structured output"
```

---

## Task 7: Build the high-level LLMService

**Files:**
- Create: `src/services/llm/llm_service.py`
- Test: `tests/services/llm/test_llm_service.py`
- Modify: `src/services/llm/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from unittest.mock import AsyncMock

from src.services.llm.llm_service import LLMService
from src.services.llm.protocol import LLMCompletion, LLMMessage
from pydantic import BaseModel


class Answer(BaseModel):
    text: str

    @classmethod
    def fallback(cls):
        return cls(text="")


@pytest.mark.asyncio
async def test_chat_completion_forwards_to_client():
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(content="hello")
    service = LLMService(client=client)

    result = await service.chat_completion("user prompt")
    assert result == "hello"
    client.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_structured_forwards_to_client():
    client = AsyncMock()
    client.generate_structured.return_value = Answer(text="ok")
    service = LLMService(client=client)

    result = await service.generate_structured("do it", Answer)
    assert result.text == "ok"
    client.generate_structured.assert_awaited_once()
```

- [ ] **Step 2: Implement LLMService**

```python
from typing import AsyncIterator, TypeVar

from pydantic import BaseModel

from src.core.config import LLM_PROVIDER, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS
from src.services.llm.config import get_model_mapping
from src.services.llm.gemini_provider import GeminiProvider
from src.services.llm.openrouter_provider import OpenRouterProvider
from src.services.llm.protocol import LLMClient, LLMMessage, LLMStreamChunk

T = TypeVar("T", bound=BaseModel)


def create_llm_client(
    provider: str | None = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
) -> LLMClient:
    provider = provider or LLM_PROVIDER
    if provider == "gemini":
        return GeminiProvider(
            default_temperature=temperature,
            default_max_tokens=max_tokens,
            default_timeout=timeout_seconds,
        )
    return OpenRouterProvider(
        default_temperature=temperature,
        default_max_tokens=max_tokens,
        default_timeout=timeout_seconds,
    )


class LLMService:
    """High-level LLM API used by routers and services."""

    def __init__(self, client: LLMClient | None = None):
        self._client = client or create_llm_client()

    async def chat_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        completion = await self._client.complete(
            messages=messages,
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        return completion.content

    async def stream_chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        return await self._client.stream(
            messages=messages,
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> T:
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        return await self._client.generate_structured(
            messages=messages,
            response_model=response_model,
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    async def grade_text_answer(
        self,
        question: str,
        user_answer: str,
        correct_answer: str | None,
        max_marks: int,
        model_key: str = "quiz",
    ) -> dict:
        from src.core.models import TextAnswerGrading

        system_prompt = (
            "You are an exam grader. Evaluate the student's answer and return ONLY JSON."
        )
        prompt = f"""Question: {question}

Correct answer: {correct_answer or "N/A"}

Student answer: {user_answer}

Max marks: {max_marks}

Return JSON with keys: is_correct (bool), marks_awarded (float), max_marks (int), feedback (string), topic (string or null)."""

        result = await self.generate_structured(
            prompt=prompt,
            response_model=TextAnswerGrading,
            system_prompt=system_prompt,
            model_key=model_key,
            temperature=0.1,
        )
        return result.model_dump()
```

- [ ] **Step 3: Update `src/services/llm/__init__.py`**

```python
from src.services.llm.protocol import LLMClient, LLMCompletion, LLMMessage, LLMStreamChunk
from src.services.llm.llm_service import LLMService, create_llm_client
from src.services.llm.structured_output import StructuredOutputParser, parse_json_text

__all__ = [
    "LLMClient",
    "LLMCompletion",
    "LLMMessage",
    "LLMStreamChunk",
    "LLMService",
    "create_llm_client",
    "StructuredOutputParser",
    "parse_json_text",
]
```

- [ ] **Step 4: Add `TextAnswerGrading` model to `src/core/models.py`**

Append after `MockTestAnalysisResponse`:

```python
class TextAnswerGrading(BaseModel):
    is_correct: bool
    marks_awarded: float
    max_marks: int
    feedback: str
    topic: Optional[str] = None

    @classmethod
    def fallback(cls):
        return cls(
            is_correct=False,
            marks_awarded=0.0,
            max_marks=1,
            feedback="Unable to grade this answer automatically.",
            topic=None,
        )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/services/llm/test_llm_service.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/services/llm/llm_service.py src/services/llm/__init__.py src/core/models.py tests/services/llm/test_llm_service.py
git commit -m "feat(llm): add high-level LLMService and TextAnswerGrading model"
```

---

## Task 8: Add fallback classmethods to response schemas

**Files:**
- Modify: `src/core/models.py`

- [ ] **Step 1: Add fallback methods to existing response models**

For each model used in structured generation, add a `fallback` classmethod.

Replace `UnitAnalysis` class body with fallback:

```python
class UnitAnalysis(BaseModel):
    unit_name: str
    weightage_percentage: float
    important_topics: List[str]
    difficulty_level: str
    recommendation: str

    @classmethod
    def fallback(cls):
        return cls(
            unit_name="General",
            weightage_percentage=0.0,
            important_topics=[],
            difficulty_level="Medium",
            recommendation="Review uploaded documents and retry.",
        )
```

Replace `QuestionPattern` class body with fallback:

```python
class QuestionPattern(BaseModel):
    question_type: str
    marks_distribution: Dict[str, int]
    frequency: int
    examples: List[str]

    @classmethod
    def fallback(cls):
        return cls(
            question_type="unknown",
            marks_distribution={},
            frequency=0,
            examples=[],
        )
```

Append fallback to `QuestionPaperAnalysisResponse`:

```python
    @classmethod
    def fallback(cls):
        return cls(
            analysis_id="",
            overall_summary="Unable to generate detailed analysis due to API response issues.",
            focus_areas=["Review your uploaded documents", "Ensure PDFs contain readable text"],
            unit_wise_analysis=[UnitAnalysis.fallback()],
            question_patterns=[QuestionPattern.fallback()],
            sample_questions=["Sample question 1?"],
            preparation_strategy="Retry analysis with better quality documents.",
            created_at=datetime.now(timezone.utc),
        )
```

Append fallback to `MockTestQuestion`:

```python
    @classmethod
    def fallback(cls):
        return cls(
            id="",
            type="mcq",
            question="Unable to generate question.",
            options=[],
            correctAnswer=None,
            marks=1,
        )
```

Append fallback to `AnswerFeedback`:

```python
    @classmethod
    def fallback(cls):
        return cls(
            question_id="",
            question="",
            user_answer="",
            correct_answer=None,
            is_correct=None,
            feedback="Unable to grade this answer.",
            marks_awarded=0.0,
            max_marks=1,
        )
```

- [ ] **Step 2: Run existing model tests**

```bash
pytest tests/ -k "not test_main" --ignore=tests/test_main.py -q
```

Expected: existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add src/core/models.py
git commit -m "feat(models): add fallback constructors for structured output schemas"
```

---

## Task 9: Migrate `llm_service.py` to LLMService

**Files:**
- Modify: `src/services/llm_service.py`
- Test: `tests/services/test_llm_service_integration.py` (new)

- [ ] **Step 1: Rewrite `llm_service.py`**

Replace the entire file with:

```python
from typing import Dict, List, Optional

from src.services.llm import LLMService
from src.services.vector_store import VectorStore
from src.services.bm25_index import BM25IndexService
from src.services.query_engine import QueryEngine

# Reusable service instances
vector_store = VectorStore()
bm25_service = BM25IndexService()
query_engine = QueryEngine(vector_store, bm25_service)
llm_service = LLMService()

_SYSTEM_PROMPT = """You are an AI tutor. 
- Provide a clear, concise, and well-structured answer.  
- Focus on key points that are important for exams.  
- Avoid unnecessary introductions—start directly with the answer.  
- If necessary, break down complex ideas into simpler explanations."""

_RAG_SYSTEM_PROMPT = """You are an AI tutor helping a student prepare for exams. Answer the following question STRICTLY based on the provided document excerpts.

Rules:
- Start directly with the answer. No introductions like "Based on the documents..."
- Cite sources inline using [index] format. Example: "The photoelectric effect [1] demonstrates that light behaves as particles."
- Focus on key points important for exams. Break down complex ideas simply.
- If the context doesn't contain enough information, say: "I don't have enough information about that in your uploaded documents."
- Do not make up information not present in the excerpts."""


async def ask_question(
    question: str,
    pdf_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    subject: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user_id: str | None = None,
    stream: bool = False,
):
    """Ask a question using the multi-document RAG system."""
    if pdf_id and not doc_ids:
        doc_ids = [pdf_id]

    context = ""
    sources = []
    chunks = []

    if user_id:
        context, sources, chunks = await query_engine.query(
            user_id=user_id,
            question=question,
            doc_ids=doc_ids,
            subject=subject,
            tags=tags,
            top_k=5,
        )

    if context:
        sources_list = "\n".join(
            f"[{i + 1}] {s['doc_name']}" + (f", {s['section']}" if s.get("section") else "") +
            (f", Page {s['page']}" if s.get("page") else f", {s['locator']}" if s.get("locator") else "")
            for i, s in enumerate(sources)
        )
        prompt = f"""{sources_list}

Document Excerpts:
{context}

Question: {question}

Answer (with inline citations):"""
        system_prompt = _RAG_SYSTEM_PROMPT
    else:
        prompt = f"**Question:** {question}\n\n**Exam-Focused Answer:**"
        system_prompt = _SYSTEM_PROMPT

    if stream:
        async def _stream():
            full_text = ""
            async for chunk in await llm_service.stream_chat(
                prompt=prompt,
                system_prompt=system_prompt,
                model_key="chat",
            ):
                if chunk.is_finished:
                    break
                full_text += chunk.content
                data = {"response": chunk.content, "done": False}
                if not context:
                    data.pop("context", None)
                yield data

            # Persist context/sources in the first chunk for RAG
            async def _full_stream():
                if context:
                    yield {"context": context, "sources": sources}
                async for item in _stream():
                    yield item
                yield {"response": "", "done": True}

            return _full_stream()

        return _stream()

    answer = await llm_service.chat_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        model_key="chat",
    )
    return {
        "answer": answer.strip(),
        "sources": sources or None,
        "context": context or None,
    }
```

- [ ] **Step 2: Write an integration-style test with mocked LLMService**

```python
import pytest
from unittest.mock import AsyncMock, patch

from src.services import llm_service as llm_module


@pytest.mark.asyncio
async def test_ask_question_without_context():
    with patch.object(llm_module, "llm_service") as mock_service:
        mock_service.chat_completion.return_value = "  The answer is 42.  "
        result = await llm_module.ask_question("what is the answer?", user_id=None, stream=False)
        assert result["answer"] == "The answer is 42."
        assert result["sources"] is None
        assert result["context"] is None
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/services/test_llm_service_integration.py -v
```

Expected: tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/services/llm_service.py tests/services/test_llm_service_integration.py
git commit -m "feat(llm): migrate chat Q&A to provider-agnostic LLMService"
```

---

## Task 10: Migrate question paper analysis to structured output

**Files:**
- Modify: `src/services/question_paper_analysis_service.py`
- Test: `tests/services/test_question_paper_analysis.py` (new or modify existing)

- [ ] **Step 1: Read the current analysis service fully**

Use `Read` on `src/services/question_paper_analysis_service.py` to capture the full prompt and response mapping before rewriting.

- [ ] **Step 2: Rewrite the service**

```python
from datetime import datetime, timezone
from typing import List

from fastapi import HTTPException

from src.core.models import QuestionPaperAnalysisResponse
from src.services.llm import LLMService

llm_service = LLMService()


async def analyze_question_papers(
    syllabus_content: str,
    question_papers_content: List[str],
    analysis_id: str,
) -> QuestionPaperAnalysisResponse:
    if not syllabus_content.strip():
        raise HTTPException(status_code=400, detail="Syllabus content is empty")

    combined_question_papers = "\n\n---NEW QUESTION PAPER---\n\n".join(question_papers_content)

    system_prompt = (
        "You are an expert academic analyst. Analyze the syllabus and previous year question papers. "
        "Respond only with valid JSON matching the required schema."
    )

    prompt = f"""Analyze the following syllabus and previous year question papers.

SYLLABUS:
{syllabus_content[:2500]}

QUESTION PAPERS:
{combined_question_papers[:4500]}

Provide a structured analysis including overall summary, focus areas, unit-wise analysis, question patterns, sample questions, and preparation strategy."""

    try:
        result = await llm_service.generate_structured(
            prompt=prompt,
            response_model=QuestionPaperAnalysisResponse,
            system_prompt=system_prompt,
            model_key="analysis",
            temperature=0.2,
        )
        result.analysis_id = analysis_id
        result.created_at = datetime.now(timezone.utc)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI analysis service unavailable: {exc!s}",
        )
```

- [ ] **Step 3: Update the analysis router to pass the analysis_id**

If `analysis_router.py` currently calls `gemini_service.analyze_question_papers`, replace it with:

```python
from src.services.question_paper_analysis_service import analyze_question_papers

# Inside the endpoint:
analysis = await analyze_question_papers(
    syllabus_content=syllabus_content,
    question_papers_content=question_papers_content,
    analysis_id=str(uuid.uuid4()),
)
return analysis
```

- [ ] **Step 4: Add a test with mocked LLMService**

```python
import pytest
from unittest.mock import AsyncMock, patch

from src.services.question_paper_analysis_service import analyze_question_papers
from src.core.models import QuestionPaperAnalysisResponse


@pytest.mark.asyncio
async def test_analyze_question_papers_success():
    fake = QuestionPaperAnalysisResponse.fallback()
    fake.analysis_id = "abc"
    fake.overall_summary = "test summary"

    with patch("src.services.question_paper_analysis_service.llm_service") as mock:
        mock.generate_structured.return_value = fake
        result = await analyze_question_papers("syllabus", ["paper1"], "abc")
        assert result.analysis_id == "abc"
        assert result.overall_summary == "test summary"
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/services/test_question_paper_analysis.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/services/question_paper_analysis_service.py tests/services/test_question_paper_analysis.py
git commit -m "feat(analysis): migrate question paper analysis to structured LLM output"
```

---

## Task 11: Migrate mock test generation and grading to structured output

**Files:**
- Modify: `src/services/mock_test_service.py`
- Test: `tests/services/test_mock_test_service.py` (new)

- [ ] **Step 1: Read the current mock_test_service fully**

Use `Read` on `src/services/mock_test_service.py` with offsets to capture all relevant prompt/grading logic.

- [ ] **Step 2: Extract a structured prompt function**

Create a new helper in `src/services/mock_test_service.py` (or a new `src/services/mock_test_prompts.py`):

```python
def build_mock_test_prompt(
    syllabus_content: str,
    question_papers_content: List[str],
    notes_content: str,
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    focus_topics: List[str] | None,
    weak_topics: List[str] | None,
    subject: str | None,
    web_examples: str,
) -> str:
    combined_question_papers = "\n\n---PREVIOUS PAPER---\n\n".join(question_papers_content)
    mcq_marks = 2
    text_marks = max(5, (total_marks - num_mcq * mcq_marks) // num_text) if num_text > 0 else 5

    focus_section = ""
    if focus_topics:
        focus_section = f"\nFOCUS TOPICS:\n{', '.join(focus_topics)}"

    weak_section = ""
    if weak_topics:
        weak_section = f"\nWEAK TOPICS:\n{', '.join(weak_topics)}"

    return f"""You are an expert exam paper setter.

INSTRUCTIONS:
1. Analyze the syllabus to understand course structure and learning outcomes.
2. Study previous year question papers to identify patterns, marks distribution, frequently asked topics, and difficulty progression.
3. Generate questions relevant to the syllabus and matching previous paper patterns.
{focus_section}{weak_section}
{f"\nSUBJECT: {subject}\n" if subject else ""}
SYLLABUS (PRIMARY REFERENCE):
{syllabus_content[:3000]}

STUDY NOTES:
{notes_content[:2000] if notes_content else "No additional notes provided"}

PREVIOUS YEAR QUESTION PAPERS:
{combined_question_papers[:4000]}

{web_examples}

REQUIREMENTS:
- Generate {num_mcq} MCQ questions worth {mcq_marks} marks each. Each MCQ must have 4 options and one correct answer.
- Generate {num_text} descriptive/text questions worth {text_marks} marks each.
- Difficulty level: {difficulty_level}
- Total marks: {total_marks}
- All questions must be traceable to syllabus topics.
- Follow question patterns from previous papers.

Respond with ONLY valid JSON matching the required schema."""
```

- [ ] **Step 3: Replace `_generate_mock_test_with_gemini`**

Replace the existing 250-line function with:

```python
from src.core.models import MockTestResponse, MockTestQuestion
from src.services.llm import LLMService

llm_service = LLMService()


async def _generate_mock_test_with_gemini(
    syllabus_content: str,
    question_papers_content: List[str],
    notes_content: str,
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    focus_topics: Optional[List[str]] = None,
    weak_topics: Optional[List[str]] = None,
    subject: Optional[str] = None,
    web_examples: str = "",
) -> MockTestResponse:
    from src.core.models import MockTestResponse

    prompt = build_mock_test_prompt(
        syllabus_content, question_papers_content, notes_content,
        num_mcq, num_text, total_marks, difficulty_level,
        focus_topics, weak_topics, subject, web_examples,
    )

    system_prompt = (
        "You are an expert exam paper setter. Respond only with valid JSON matching the required schema."
    )

    result = await llm_service.generate_structured(
        prompt=prompt,
        response_model=MockTestResponse,
        system_prompt=system_prompt,
        model_key="quiz",
        temperature=0.3,
    )

    # Ensure each question has a deterministic id
    for i, q in enumerate(result.questions):
        q.id = q.id or f"q-{i + 1}"

    result.title = result.title or "Mock Test"
    result.total_marks = result.total_marks or total_marks
    result.time_limit = result.time_limit or _calculate_time_limit(total_marks, num_mcq, num_text)
    result.difficulty_level = result.difficulty_level or difficulty_level
    return result
```

- [ ] **Step 4: Replace text-answer grading with LLMService**

Find the grading logic in `mock_test_service.py` (likely around line 400+) and replace the Gemini grading block with:

```python
from src.services.llm import LLMService

llm_service = LLMService()


async def _grade_text_answer(
    question: str,
    user_answer: str,
    correct_answer: str | None,
    max_marks: int,
) -> dict:
    return await llm_service.grade_text_answer(
        question=question,
        user_answer=user_answer,
        correct_answer=correct_answer,
        max_marks=max_marks,
        model_key="quiz",
    )
```

Then replace the inline grading call with `await _grade_text_answer(...)`.

- [ ] **Step 5: Write a test for mock test generation with mocked LLMService**

```python
import pytest
from unittest.mock import AsyncMock, patch

from src.services.mock_test_service import _generate_mock_test_with_gemini
from src.core.models import MockTestResponse, MockTestQuestion


@pytest.mark.asyncio
async def test_generate_mock_test_structured():
    fake = MockTestResponse(
        test_id="t1",
        title="Test",
        questions=[MockTestQuestion(id="q1", type="mcq", question="What is 2+2?", options=["3", "4"], correctAnswer="4", marks=2)],
        total_marks=50,
        time_limit=60,
        created_at=datetime.now(timezone.utc),
        user_id="u1",
    )

    with patch("src.services.mock_test_service.llm_service") as mock:
        mock.generate_structured.return_value = fake
        result = await _generate_mock_test_with_gemini(
            "syllabus", ["paper"], "notes", 1, 0, 50, "medium"
        )
        assert result.title == "Test"
        assert len(result.questions) == 1
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/services/test_mock_test_service.py -v
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/services/mock_test_service.py tests/services/test_mock_test_service.py
git commit -m "feat(mock-tests): migrate generation and grading to structured LLM output"
```

---

## Task 12: Clean up old gemini_service usage

**Files:**
- Modify: `src/services/gemini_service.py`
- Modify: `src/routers/analysis_router.py` if it still imports `gemini_service`

- [ ] **Step 1: Reduce `gemini_service.py` to a text-extraction helper**

Replace the file with:

```python
from pathlib import Path
from pypdf import PdfReader


async def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a local PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)
```

- [ ] **Step 2: Update imports in `mock_test_service.py`**

Replace:
```python
from src.services.gemini_service import gemini_service
```
With:
```python
from src.services.gemini_service import extract_text_from_pdf
```

Replace all `await gemini_service.extract_text_from_pdf(...)` with `await extract_text_from_pdf(...)`.

- [ ] **Step 3: Update `analysis_router.py`**

If it imports `gemini_service`, replace the call with:
```python
from src.services.gemini_service import extract_text_from_pdf
from src.services.question_paper_analysis_service import analyze_question_papers
```

- [ ] **Step 4: Run existing tests**

```bash
pytest tests/ -k "not test_main" --ignore=tests/test_main.py -q
```

Expected: existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/services/gemini_service.py src/services/mock_test_service.py src/routers/analysis_router.py
git commit -m "refactor(gemini): keep only PDF text extraction helper"
```

---

## Task 13: Full regression test run

**Files:**
- All touched files

- [ ] **Step 1: Run all backend tests**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source .venv/bin/activate
pytest tests/ -q
```

Expected: all tests pass. If `test_main.py` fails due to missing MongoDB, run with `--ignore=tests/test_main.py` and document it as a pre-existing environment issue.

- [ ] **Step 2: Run compile check**

```bash
python -m py_compile $(find src -name '*.py')
```

Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "test(track-a): full regression test pass for LLM abstraction track"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every section of the design doc maps to one or more tasks above.
- [ ] **No placeholders:** Every task has concrete code, exact commands, and expected output.
- [ ] **Type consistency:** `LLMMessage`, `LLMCompletion`, `LLMStreamChunk`, `generate_structured` signatures match across protocol and providers.
- [ ] **Backward compatibility:** Existing request/response models unchanged; only internal AI layer changes.
- [ ] **Test coverage:** New unit tests cover parser, both providers, LLMService, and migrated services.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-14-track-a-llm-abstraction-plan.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach would you like?
