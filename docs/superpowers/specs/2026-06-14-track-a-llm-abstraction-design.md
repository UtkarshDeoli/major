# Track A: LLM Provider Abstraction + Structured Outputs

## Goal

Make the Orbit backend provider-agnostic and replace brittle, regex-based JSON parsing with validated, typed, retryable structured outputs. Keep all existing API request/response contracts unchanged so the frontend continues to work.

## Background

The current backend is tightly coupled to the legacy `google-generativeai` SDK and a single hard-coded model (`gemini-2.5-flash`). The README promises an OpenRouter-based stack with free models, but the implementation does not match. In addition, analysis and mock-test generation parse AI responses with fragile string slicing (`response_text.find('{')`, `response_text.rfind('}')`), which frequently fails when models wrap JSON in markdown fences or explanatory text.

## Scope

This track covers:

1. Introducing a provider-agnostic `LLMClient` protocol.
2. Implementing two providers: `OpenRouterProvider` (primary) and `GeminiProvider` (modern `google-genai` SDK).
3. Adding a `StructuredOutputParser` for Pydantic-based validation, retry, and typed fallback.
4. Building `LLMService` high-level methods used by routers and services.
5. Migrating `llm_service.py`, `question_paper_analysis_service.py`, and `mock_test_service.py` to the new layer.
6. Adding unit tests for the new components.

Out of scope for this track:

- Auth/user_id changes (Track B).
- Repository split of `data_store.py` (Track B).
- Persistent BM25, S3 storage, rate limiting, or observability (Track C).

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                         Routers                              │
│  question_router  │  analysis_router  │  mock_test_router   │
└───────────────┬───────────────┬───────────────┬───────────────┘
                │               │               │
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLMService                              │
│  - chat_completion(prompt, stream=False)                     │
│  - generate_structured(prompt, response_model)               │
│  - grade_answer(...)                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌───────────────────┐           ┌───────────────────┐
│ OpenRouterProvider │           │  GeminiProvider   │
│  (openai SDK)      │           │  (google-genai)   │
└───────────────────┘           └───────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ▼
            ┌───────────────────────┐
            │ StructuredOutputParser │
            │  - json extraction     │
            │  - pydantic validation   │
            │  - retry + fallback    │
            └───────────────────────┘
```

## Provider Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `gemini` |
| `OPENROUTER_API_KEY` | — | Required when provider is `openrouter` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OR_MODEL_CHAT` | `meta-llama/llama-3.3-70b-instruct` | Chat / Q&A model |
| `OR_MODEL_ANALYSIS` | `deepseek/deepseek-r1-0528` | Question-paper analysis model |
| `OR_MODEL_QUIZ` | `stepfun/step-3.5-flash` | Mock-test generation model |
| `OR_MODEL_FALLBACK` | `openrouter/free` | Fallback model |
| `GEMINI_API_KEY` | — | Required when provider is `gemini` |
| `GEMINI_MODEL_CHAT` | `gemini-2.5-flash` | Chat / Q&A model |
| `GEMINI_MODEL_ANALYSIS` | `gemini-2.5-flash` | Analysis model |
| `GEMINI_MODEL_QUIZ` | `gemini-2.5-flash` | Quiz generation model |
| `GEMINI_MODEL_FALLBACK` | `gemini-2.5-flash` | Fallback model |
| `LLM_TEMPERATURE` | `0.3` | Default generation temperature |
| `LLM_MAX_TOKENS` | `8192` | Default max output tokens |
| `LLM_TIMEOUT_SECONDS` | `120` | Default request timeout |
| `LLM_MAX_RETRIES` | `3` | Max retries for structured output parsing |

## Data Models

### Internal Protocol

```python
from typing import Protocol, AsyncIterator
from dataclasses import dataclass

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
    async def complete(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletion: ...

    async def stream(
        self,
        messages: list[LLMMessage],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[LLMStreamChunk]: ...

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        model_key: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> BaseModel: ...
```

### Response Schemas (new / reused)

Reuse existing response schemas in `src.core.models`:

- `QuestionPaperAnalysisResponse`
- `UnitAnalysis`
- `QuestionPattern`
- `MockTestQuestion`
- `MockTestResponse`
- `AnswerFeedback`
- `MockTestAnalysisResponse`

Add one internal schema for LLM-graded text answers if needed:

```python
class TextAnswerGrading(BaseModel):
    is_correct: bool
    marks_awarded: float
    max_marks: int
    feedback: str
    topic: str | None = None
```

## Structured Output Strategy

### Gemini

Use native JSON schema support:

```python
from google.genai import types

config = types.GenerateContentConfig(
    temperature=0.1,
    max_output_tokens=8192,
    response_mime_type="application/json",
    response_json_schema=response_model.model_json_schema(),
)
```

### OpenRouter

Most free models do not reliably support native JSON schema. Use a two-stage strategy:

1. Send a prompt that includes the JSON schema and asks for JSON-only output.
2. Parse the response with `StructuredOutputParser`:
   - Strip markdown code fences.
   - Extract the largest JSON object.
   - Validate with Pydantic.
   - Retry up to `LLM_MAX_RETRIES` with exponential backoff if validation fails.
3. If all retries fail, return a typed fallback instance built by the response model's `fallback()` class method.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Provider API key missing | Raise clear `ValueError` at startup; healthcheck remains OK but LLM endpoints return 503. |
| Provider returns empty / blocked | Treat as parse failure; retry; if exhausted, return typed fallback. |
| Provider rate limit / timeout | Return HTTP 503/504 with a clean detail message. Do not leak raw traceback to client. |
| JSON parse / validation failure | Retry with slightly different prompt (e.g., "respond with only raw JSON, no markdown"). |
| Streaming error | Yield an error chunk in NDJSON; router logs but keeps connection alive. |

## Files to Create

| File | Responsibility |
|------|--------------|
| `src/services/llm/protocol.py` | `LLMMessage`, `LLMCompletion`, `LLMStreamChunk`, `LLMClient` protocol. |
| `src/services/llm/config.py` | Provider/model configuration, `LLMProviderSettings`. |
| `src/services/llm/openrouter_provider.py` | `OpenRouterProvider` implementation using `AsyncOpenAI`. |
| `src/services/llm/gemini_provider.py` | `GeminiProvider` implementation using `google-genai` async client. |
| `src/services/llm/structured_output.py` | JSON extraction, Pydantic validation, retry, typed fallback. |
| `src/services/llm/llm_service.py` | High-level `LLMService` with `chat`, `generate_structured`, `grade_text_answer`. |
| `src/services/llm/__init__.py` | Public exports. |

## Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Replace `google-generativeai` with `google-genai>=2.8.0`; add `openai>=1.12.0`. |
| `src/core/config.py` | Add LLM provider/model env vars. |
| `src/services/llm_service.py` | Replace Gemini calls with `LLMService.chat_completion` and `LLMService.stream`. |
| `src/services/gemini_service.py` | Deprecate / remove after migrating PDF text extraction; if still needed, keep a thin text-extraction helper. |
| `src/services/question_paper_analysis_service.py` | Use `LLMService.generate_structured(QuestionPaperAnalysisResponse)`. |
| `src/services/mock_test_service.py` | Use `LLMService.generate_structured` for test generation and text-answer grading. |
| `src/core/models.py` | Add `fallback()` classmethods to response schemas where missing. |

## Backward Compatibility

- All existing FastAPI request/response models keep identical field names and types.
- Existing env vars (`GEMINI_API_KEY`) continue to work when `LLM_PROVIDER=gemini`.
- Streaming endpoint still returns `application/x-ndjson` chunks in the same shape.
- No database schema changes.

## Testing Strategy

- Unit test `StructuredOutputParser` against markdown-wrapped JSON, invalid JSON, and fallback behavior.
- Unit test `OpenRouterProvider` and `GeminiProvider` with mocked HTTP / SDK responses.
- Unit test `LLMService` chat and structured generation.
- Integration-style test for the migrated `mock_test_service` and `question_paper_analysis_service` using mocked `LLMService`.
- Run existing `pytest` suite to ensure no regressions.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OpenRouter free models are unreliable | Keep Gemini as an alternate provider; add fallback model mapping. |
| Structured JSON still fails after retries | Provide typed fallback so users see a graceful result instead of 500. |
| Streaming migration breaks frontend | Keep exact NDJSON chunk shape; add tests. |
| Removing old `gemini_service.py` breaks PDF text extraction | Move text extraction to `document_processor.py` or keep a thin helper. |

## Success Criteria

1. `LLM_PROVIDER=openrouter` works end-to-end for `/questions/ask`, `/analysis/question-papers`, and `/mock-tests/generate`.
2. `LLM_PROVIDER=gemini` still works with the new SDK.
3. Analysis and mock-test endpoints return valid JSON matching their Pydantic schemas ≥99% of the time (measured via tests).
4. Streaming `/questions/ask/stream` is truly async and yields chunks without blocking the event loop.
5. All new code has unit tests; existing tests still pass.
