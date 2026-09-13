# Report: Refactoring `Rag_Legal_Assitant` for extensibility

## 1. Overview

`Rag_Legal_Assitant` is a Vietnamese legal Q&A chatbot API built on a RAG architecture (FastAPI + FAISS + an OpenAI-compatible LLM), developed for the R2AI 2026 competition (see `README.md`). This report records:

- The architecture's state before the refactor and the specific problems found.
- What changed, why, and how it was verified.
- How to extend the system under the new architecture (adding an LLM provider / vector store / tool).
- Known issues **deliberately left out of scope** this time.

## 2. State before the refactor

Reading through the entire codebase (`api/`, `services/`, `pipeline/`) showed a working system with 8 groups of problems blocking extensibility:

| # | Problem | Consequence |
| - | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A | No interface for LLM / Embedding / Reranker / Vector store / Tool | Switching provider (e.g. to Anthropic, Qdrant, adding a new tool) forces changes to core code (`RAGPipeline.process()`) |
| B | Scattered config, repeated `os.getenv` across 3 files | `.env.example` declares `RERANK_*` but the code reads `RERANKER_*` → **reranker never activates** even with correct config per the docs |
| C | `ChatService`/`SearchService` are "God objects" | 1 class carries 3-5 unrelated responsibilities (chat + embedding + rerank; FAISS + citation parsing + 2 search strategies) |
| D | Tool-calling hardcoded with `if/else` | Adding a 2nd tool requires editing the orchestration loop inside `RAGPipeline` |
| E | Prompts mixed into orchestration logic | 3 near-duplicate Vietnamese prompt blocks scattered inside `RAGPipeline.py` |
| F | No schema for the data contract between modules | Caused a real bug: the final `answer/done` event in the stream **was missing the `sources` key**, so the non-stream response (`stream=false`) always returned `sources: []` |
| G | `pipeline/` (offline crawl/chunk/embed) and `services/` are 2 separate worlds | Duplicated logic for building the OpenAI embedding client in 2 places |
| H | No tests anywhere in the repo | Refactoring had no regression safety net |

There was also 1 unsafe default value: `CHAT_BASE_URL` defaulted to an internal IP (`http://10.9.3.241:30040/v1`) whenever it was left blank in `.env`.

## 3. Architecture after the refactor

```
core/        → Protocols/interfaces + centralized Settings + shared models + prompt text (depends on nothing below it)
providers/   → Concrete implementations of each interface (OpenAI LLM/Embedding, vLLM Reranker, FAISS VectorStore)
tools/       → ToolRegistry (dict-based) + the existing tool (search_referenced_document)
services/    → Pure business logic: search.py (retrieval) + rag_pipeline.py (orchestration) — depends only on interfaces
api/app.py   → Composition root: the ONLY place that builds concrete providers and injects them into services
pipeline/    → Offline scripts (crawl/chunk/embed), reuses OpenAIEmbeddingProvider instead of building its own client
tests/       → pytest, runs fully offline using fake/in-memory implementations
```

Principle: `core` ← `providers`/`tools` ← `services` ← `api` — one direction, no back edges, so an import cycle is impossible.

### 3.1. New interfaces (`core/interfaces.py`)

| Interface | Main method | Concrete implementation |
| --------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `LLMProvider` | `chat(messages, tools=, response_format=, stream=)` | `providers/openai_llm.py::OpenAILLMProvider` |
| `EmbeddingProvider` | `embed(text) -> list[float]` | `providers/openai_embedding.py::OpenAIEmbeddingProvider` |
| `Reranker` | `rerank(query, documents) -> list[float]` | `providers/reranker.py::VLLMReranker` / `NullReranker` |
| `VectorStore` | `search`, `search_subset`, `chunk_id_for`, `faiss_id_for`, `total` | `providers/faiss_store.py::FaissVectorStore` |
| `Tool` | `name`, `schema`, `execute(args, question=)` | `tools/doc_ref_tool.py::SearchReferencedDocumentTool` |

Uses `typing.Protocol` (structural typing) instead of an abstract base class — any class implementing the right method signature automatically satisfies the interface, no inheritance needed.

### 3.2. Splitting up the "God objects"

- `services/Chat.py` (chat + embedding + rerank) → split into 3 independent providers (`OpenAILLMProvider`, `OpenAIEmbeddingProvider`, `VLLMReranker`).
- `services/Search.py` → split into `SemanticSearchService` (semantic search + rerank) and `DocRefSearchService` (lookup by document citation), both depending only on interfaces, with no direct `faiss`/SDK access.
- `services/RAGPipeline.py` → `services/rag_pipeline.py`, now holds only orchestration logic; prompts moved to `core/prompts.py`, tool dispatch moved to `ToolRegistry`.

### 3.3. Centralized config (`core/config.py`)

Uses `pydantic-settings`. `CHAT_*`/`EMBEDDING_*` (6 vars) are **required, with no default** — if any is missing, the server reports a clear error immediately at startup instead of silently falling back to the old internal IP. Verified in practice:

```
$ python -c "from api.app import build_pipeline; build_pipeline()"
pydantic_core._pydantic_core.ValidationError: 3 validation errors for ChatSettings
CHAT_BASE_URL   Field required [type=missing]
CHAT_API_KEY    Field required [type=missing]
CHAT_MODEL_NAME Field required [type=missing]
```

Other operational parameters (`MAX_CONTEXT_CHUNKS`, `MAX_TOOL_ITERATIONS`, `SEMANTIC_TOP_K`, `TOOL_SEARCH_TOP_K`, `RETRIEVAL_THRESHOLD`, `DATA_DIR`) all default to the old hardcoded values and can be overridden via env without touching the code.

### 3.4. Tool registry (`tools/registry.py`)

Replaces `if tc["function"]["name"] != "search_referenced_document": ... else: ...` with:

```python
tool = self.tool_registry.get(tc["function"]["name"])
if tool is None:
    llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Tool không được hỗ trợ."})
    continue
extra_docs = tool.execute(args, question=question)
```

Adding a new tool = write 1 class implementing `Tool` + register it with `tool_registry.register(...)` in `api/app.py::build_pipeline()`. **No need to touch `rag_pipeline.py`.**

## 4. Bugs fixed

| Bug | Before | After |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RERANK_*` vs `RERANKER_*` | `.env.example` declares `RERANK_*`, code reads `RERANKER_*` → rerank never turns on | `core/config.py` reads `RERANK_*` correctly, matching `.env.example`. Regression test (`test_reranker_stays_disabled_with_legacy_env_var_name`) ensures the old variable name **does not** accidentally re-enable it |
| Hardcoded internal IP as default | `CHAT_BASE_URL` defaults to `http://10.9.3.241:30040/v1` when left blank | No more default — required to declare, fails fast if missing |
| `sources` missing from non-stream response | The final `answer/done` event only had `text`+`citations`, so `stream=false` always returned `sources: []` | The `answer/done` event now also includes `sources: context_docs` (data already existed, just wasn't being included) — regression test `test_answer_done_event_includes_sources` |
| `/no_think` wasn't actually running | README described a `_with_no_think()` helper that **did not exist** in the code | Actually implemented in `core/prompts.py::with_no_think()`, wired into both LLM calls (sub-query + answer), without mutating the original conversation history — test `test_with_no_think_applied_without_mutating_history` |

## 5. Test coverage

47 tests, running fully **offline** (no need to load the 3.2GB FAISS index or a real LLM endpoint):

```
tests/test_config.py         — fail-fast on missing required env vars, reranker enable/disable, overriding params via env
tests/test_providers.py      — LLM/Embedding provider (mocked OpenAI client), VLLMReranker (mocked requests.post),
                                FaissVectorStore (builds a real small FAISS index in memory — no download needed)
tests/test_tool_registry.py  — register/get/schemas(), mapping tool params → DocRefSearchService
tests/test_search_service.py — threshold filter, rerank resort, parsing Vietnamese citations, doc_ref_search
                                (exact match / fuzzy fallback / article-clause filtering)
tests/test_rag_pipeline.py   — `answer/done` event includes sources (regression), event order when a tool call happens,
                                an unknown tool doesn't crash, with_no_think doesn't mutate history, still emits
                                a final event when MAX_TOOL_ITERATIONS is reached
```

Result: `47 passed` (`pytest tests/ -q`). Also ran a manual smoke test: built a fake `data/` (small FAISS index + JSON maps), called `build_pipeline()` successfully, started a real server (`uvicorn api.app:app`) and called `GET /health` (200 OK) + `POST /chat` (stream=false) — the pipeline ran through the right flow (sub-query → retrieval → context_ready → tool_call → answer) and gracefully reported an LLM connection error since there was no real LLM endpoint in the test environment.

## 6. How to extend (under the new architecture)

- **Add a new LLM provider** (e.g. calling the Anthropic SDK directly): create `providers/anthropic_llm.py` implementing `LLMProvider.chat(...)`, change 1 line initializing `llm = ...` in `api/app.py::build_pipeline()`. No need to touch `rag_pipeline.py`.
- **Add a different vector store** (Qdrant/Milvus/pgvector): create `providers/qdrant_store.py` implementing `VectorStore`, change the `vector_store = ...` initialization line. `services/search.py` needs no changes since it only calls through the interface.
- **Add a new tool**: create a class implementing `Tool` (`core/interfaces.py`), register it with `tool_registry.register(YourTool(...))` in the composition root.
- **Add a new operational parameter**: add a field to `core/config.py::RetrievalSettings` (or the relevant settings class); no need to change logic elsewhere as long as `settings` is already injected there.

## 7. Known issues, deliberately out of scope

While writing tests, 1 more pre-existing quirk was found (not caused by the refactor, behavior kept unchanged per the agreed scope — no "silently fixing" bugs outside the 3 bugs in section 4):

- **`_extract_doc_num` (services/search.py)**: the regex identifying document numbers only matches the uppercase part in the final group (`[A-ZĐƯƠ]+`). For document numbers with lowercase letters at the end (e.g. `QĐ-TTg`), the lowercase part gets cut off (`QĐ-TT`). Chunk matching still works correctly thanks to the fuzzy-match fallback on `doc_num`/`title` right after, but this is worth noting if new document-number formats are added later or the first-pass match is tightened.
- **Filtering by threshold before rerank** (`SemanticSearchService.semantic_search`): candidates are dropped based on the raw FAISS score before rerank, so a candidate that the reranker would have scored highly could be dropped by mistake — this behavior is kept identical to the original, and is not within the scope of the 3 approved bug fixes this time.

Both are noted directly in the code/README so they aren't forgotten if someone touches this area later.

## 8. Changes to know about when updating to the refactored version

- **New required vars**: `CHAT_BASE_URL`, `CHAT_API_KEY`, `CHAT_MODEL_NAME`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL_NAME` — anyone who already has a complete `.env` per `.env.example` is unaffected.
- **Unchanged**: the `PORT` variable, running via `python main.py`, file names/formats in `data/`, the `/chat` request shape.
- **Intentional changes**: the `stream=false` response now returns full `sources` instead of always being empty; `/no_think` is now actually sent to the LLM (previously described in docs but never ran).
- Old files removed: `services/Chat.py`, `services/Search.py`, `services/RAGPipeline.py`, `services/OpenAIExtended.py` — replaced by modules under `core/`, `providers/`, `services/search.py`, `services/rag_pipeline.py`.
- Setup: `pip install -r requirements.txt` (to run the API), add `-r requirements-pipeline.txt` if running `pipeline/`, add `-r requirements-dev.txt` if running tests.
- Or use [uv](https://docs.astral.sh/uv/): `uv sync` (installs test dependencies too); `uv run python main.py` to run the API, `uv run pytest` to run tests. See section 9.

## 9. Using uv instead of pip/venv

The project has a `pyproject.toml` declaring dependencies (kept in sync with `requirements.txt`), usable with [uv](https://docs.astral.sh/uv/):

```bash
uv sync              # create .venv + install dependencies (including the dev group: pytest, pytest-mock)
uv run python main.py   # run the API, equivalent to python main.py after activating the venv
uv run pytest            # run tests
uv add <package>         # add a new dependency, auto-updates pyproject.toml + uv.lock
```

`uv.lock` pins the exact resolved versions — commit this file so everyone installs the same set of dependencies.
