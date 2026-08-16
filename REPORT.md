# Báo cáo: Refactor kiến trúc `Rag_Legal_Assitant` để dễ mở rộng

## 1. Tổng quan

`Rag_Legal_Assitant` là API chatbot hỏi-đáp pháp luật Việt Nam theo kiến trúc RAG (FastAPI + FAISS + LLM tương thích OpenAI), phát triển cho cuộc thi R2AI 2026 (xem `README.md`). Báo cáo này ghi lại:

- Hiện trạng kiến trúc trước khi refactor và các vấn đề cụ thể phát hiện được.
- Những gì đã thay đổi, vì sao, và cách kiểm chứng.
- Cách mở rộng hệ thống theo kiến trúc mới (thêm LLM provider / vector store / tool).
- Các vấn đề đã biết nhưng **cố tình để ngoài phạm vi** lần này.

## 2. Hiện trạng trước refactor

Đọc toàn bộ mã nguồn (`api/`, `services/`, `pipeline/`) cho thấy hệ thống chạy đúng nhưng có 8 nhóm vấn đề cản trở mở rộng:

| # | Vấn đề | Hệ quả |
|---|---|---|
| A | Không có interface cho LLM / Embedding / Reranker / Vector store / Tool | Đổi provider (vd. sang Anthropic, Qdrant, thêm tool mới) bắt buộc sửa code lõi (`RAGPipeline.process()`) |
| B | Config rải rác, `os.getenv` lặp lại ở 3 file | `.env.example` khai báo `RERANK_*` nhưng code đọc `RERANKER_*` → **reranker không bao giờ kích hoạt** dù cấu hình đúng theo tài liệu |
| C | `ChatService`/`SearchService` là "God object" | 1 class gánh 3-5 trách nhiệm không liên quan (chat + embedding + rerank; FAISS + parse citation + 2 chiến lược search) |
| D | Tool-calling hardcode bằng `if/else` | Thêm tool thứ 2 phải sửa vòng lặp orchestration trong `RAGPipeline` |
| E | Prompt lẫn vào logic điều phối | 3 đoạn prompt tiếng Việt gần trùng nhau nằm rải rác trong `RAGPipeline.py` |
| F | Hợp đồng dữ liệu giữa các module không có schema | Gây bug thật: event `answer/done` cuối luồng **thiếu key `sources`**, khiến response non-stream (`stream=false`) luôn trả `sources: []` |
| G | `pipeline/` (crawl/chunk/embed offline) và `services/` là 2 thế giới tách biệt | Trùng lặp logic dựng OpenAI embedding client ở 2 nơi |
| H | Không có test nào trong repo | Refactor không có lưới an toàn hồi quy |

Ngoài ra còn 1 giá trị mặc định không an toàn: `CHAT_BASE_URL` mặc định về một IP nội bộ (`http://10.9.3.241:30040/v1`) nếu bị bỏ trống trong `.env`.

## 3. Kiến trúc sau refactor

```
core/        → Protocol/interface + Settings tập trung + shared models + prompt text (không phụ thuộc gì bên dưới)
providers/   → Cài đặt cụ thể của từng interface (OpenAI LLM/Embedding, vLLM Reranker, FAISS VectorStore)
tools/       → ToolRegistry (dict-based) + tool hiện có (search_referenced_document)
services/    → Logic nghiệp vụ thuần: search.py (retrieval) + rag_pipeline.py (orchestration) — chỉ phụ thuộc interface
api/app.py   → Composition root: nơi DUY NHẤT dựng provider cụ thể và inject vào services
pipeline/    → Script offline (crawl/chunk/embed), dùng lại OpenAIEmbeddingProvider thay vì tự dựng client riêng
tests/       → pytest, chạy hoàn toàn offline bằng fake/in-memory implementation
```

Nguyên tắc: `core` ← `providers`/`tools` ← `services` ← `api` — một chiều, không có cạnh ngược, nên không thể có import cycle.

### 3.1. Các interface mới (`core/interfaces.py`)

| Interface | Method chính | Cài đặt cụ thể |
|---|---|---|
| `LLMProvider` | `chat(messages, tools=, response_format=, stream=)` | `providers/openai_llm.py::OpenAILLMProvider` |
| `EmbeddingProvider` | `embed(text) -> list[float]` | `providers/openai_embedding.py::OpenAIEmbeddingProvider` |
| `Reranker` | `rerank(query, documents) -> list[float]` | `providers/reranker.py::VLLMReranker` / `NullReranker` |
| `VectorStore` | `search`, `search_subset`, `chunk_id_for`, `faiss_id_for`, `total` | `providers/faiss_store.py::FaissVectorStore` |
| `Tool` | `name`, `schema`, `execute(args, question=)` | `tools/doc_ref_tool.py::SearchReferencedDocumentTool` |

Dùng `typing.Protocol` (structural typing) thay vì abstract base class — bất kỳ class nào implement đúng method signature đều tự động thỏa interface, không cần kế thừa.

### 3.2. Tách "God object"

- `services/Chat.py` (chat + embedding + rerank) → tách thành 3 provider độc lập (`OpenAILLMProvider`, `OpenAIEmbeddingProvider`, `VLLMReranker`).
- `services/Search.py` → tách thành `SemanticSearchService` (semantic search + rerank) và `DocRefSearchService` (tra cứu theo trích dẫn văn bản), cả hai chỉ phụ thuộc interface, không đụng trực tiếp `faiss`/SDK.
- `services/RAGPipeline.py` → `services/rag_pipeline.py`, chỉ còn logic điều phối; prompt chuyển sang `core/prompts.py`, tool-dispatch chuyển sang `ToolRegistry`.

### 3.3. Config tập trung (`core/config.py`)

Dùng `pydantic-settings`. `CHAT_*`/`EMBEDDING_*` (6 biến) **bắt buộc, không có giá trị mặc định** — thiếu biến nào, server báo lỗi rõ ràng ngay khi khởi động thay vì âm thầm dùng IP nội bộ cũ. Đã kiểm chứng thực tế:

```
$ python -c "from api.app import build_pipeline; build_pipeline()"
pydantic_core._pydantic_core.ValidationError: 3 validation errors for ChatSettings
CHAT_BASE_URL   Field required [type=missing]
CHAT_API_KEY    Field required [type=missing]
CHAT_MODEL_NAME Field required [type=missing]
```

Các tham số vận hành khác (`MAX_CONTEXT_CHUNKS`, `MAX_TOOL_ITERATIONS`, `SEMANTIC_TOP_K`, `TOOL_SEARCH_TOP_K`, `RETRIEVAL_THRESHOLD`, `DATA_DIR`) đều có default = giá trị hardcode cũ, có thể override qua env mà không cần sửa code.

### 3.4. Tool registry (`tools/registry.py`)

Thay `if tc["function"]["name"] != "search_referenced_document": ... else: ...` bằng:

```python
tool = self.tool_registry.get(tc["function"]["name"])
if tool is None:
    llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Tool không được hỗ trợ."})
    continue
extra_docs = tool.execute(args, question=question)
```

Thêm tool mới = viết 1 class implement `Tool` + đăng ký `tool_registry.register(...)` trong `api/app.py::build_pipeline()`. **Không cần sửa `rag_pipeline.py`.**

## 4. Bug đã sửa

| Bug | Trước | Sau |
|---|---|---|
| `RERANK_*` vs `RERANKER_*` | `.env.example` khai `RERANK_*`, code đọc `RERANKER_*` → rerank không bao giờ bật | `core/config.py` đọc đúng `RERANK_*`, khớp `.env.example`. Có test hồi quy (`test_reranker_stays_disabled_with_legacy_env_var_name`) đảm bảo tên biến cũ **không** vô tình kích hoạt lại |
| IP nội bộ hardcode làm default | `CHAT_BASE_URL` mặc định `http://10.9.3.241:30040/v1` nếu bỏ trống | Không còn default — bắt buộc khai báo, fail fast nếu thiếu |
| `sources` thiếu trong response non-stream | Event `answer/done` cuối cùng chỉ có `text`+`citations`, khiến `stream=false` luôn trả `sources: []` | Event `answer/done` nay có thêm `sources: context_docs` (đã có sẵn dữ liệu, chỉ là chưa được đưa vào) — có test hồi quy `test_answer_done_event_includes_sources` |
| `/no_think` chưa thực sự chạy | README mô tả helper `_with_no_think()` nhưng **không tồn tại** trong code | Implement thật `core/prompts.py::with_no_think()`, gắn vào cả 2 lời gọi LLM (sub-query + answer), không mutate lịch sử hội thoại gốc — có test `test_with_no_think_applied_without_mutating_history` |

## 5. Test coverage

47 test, chạy hoàn toàn **offline** (không cần tải FAISS index 3.2GB hay endpoint LLM thật):

```
tests/test_config.py         — fail-fast khi thiếu env bắt buộc, reranker enable/disable, override tham số qua env
tests/test_providers.py      — LLM/Embedding provider (mock OpenAI client), VLLMReranker (mock requests.post),
                                FaissVectorStore (build 1 index FAISS thật nhỏ trong bộ nhớ — không cần tải gì)
tests/test_tool_registry.py  — register/get/schemas(), map tham số tool → DocRefSearchService
tests/test_search_service.py — threshold filter, rerank resort, parse trích dẫn tiếng Việt, doc_ref_search
                                (exact match / fuzzy fallback / lọc điều-khoản)
tests/test_rag_pipeline.py   — event `answer/done` có sources (regression), thứ tự event khi có tool-call,
                                tool lạ không crash, with_no_think không mutate lịch sử, hết MAX_TOOL_ITERATIONS
                                vẫn phát event kết thúc
```

Kết quả: `47 passed` (`pytest tests/ -q`). Đã chạy thêm smoke-test thủ công: dựng `data/` giả (FAISS index nhỏ + JSON map), gọi `build_pipeline()` thành công, khởi động server thật (`uvicorn api.app:app`) và gọi `GET /health` (200 OK) + `POST /chat` (stream=false) — pipeline chạy đúng luồng (sub-query → retrieval → context_ready → tool_call → answer) và báo lỗi kết nối LLM một cách graceful vì không có LLM endpoint thật trong môi trường test.

## 6. Cách mở rộng (theo kiến trúc mới)

- **Thêm LLM provider mới** (vd. gọi thẳng Anthropic SDK): tạo `providers/anthropic_llm.py` implement `LLMProvider.chat(...)`, đổi 1 dòng khởi tạo `llm = ...` trong `api/app.py::build_pipeline()`. Không đụng `rag_pipeline.py`.
- **Thêm vector store khác** (Qdrant/Milvus/pgvector): tạo `providers/qdrant_store.py` implement `VectorStore`, đổi dòng khởi tạo `vector_store = ...`. `services/search.py` không cần sửa vì chỉ gọi qua interface.
- **Thêm tool mới**: tạo class implement `Tool` (`core/interfaces.py`), đăng ký `tool_registry.register(YourTool(...))` trong composition root.
- **Thêm tham số vận hành mới**: thêm field vào `core/config.py::RetrievalSettings` (hoặc settings tương ứng), không cần sửa logic ở nơi khác nếu đã inject `settings` vào.

## 7. Vấn đề đã biết, cố tình để ngoài phạm vi

Trong lúc viết test, phát hiện thêm 1 quirk có sẵn từ trước (không phải do refactor gây ra, giữ nguyên hành vi theo đúng phạm vi đã thống nhất — không "sửa ngầm" các bug ngoài 3 bug ở mục 4):

- **`_extract_doc_num` (services/search.py)**: regex nhận diện số hiệu văn bản chỉ khớp phần chữ HOA ở nhóm cuối (`[A-ZĐƯƠ]+`). Với số hiệu có chữ thường ở cuối (vd. `QĐ-TTg`), phần chữ thường bị cắt mất (`QĐ-TT`). Việc match chunk vẫn đúng nhờ bước fallback fuzzy-match theo `doc_num`/`title` ngay sau đó, nhưng đây là điểm cần lưu ý nếu sau này mở rộng thêm định dạng số hiệu văn bản khác hoặc siết chặt match ở bước đầu.
- **Filter theo threshold trước khi rerank** (`SemanticSearchService.semantic_search`): candidate bị loại theo điểm FAISS thô trước khi rerank, nên có thể loại nhầm candidate mà rerank lẽ ra sẽ chấm điểm cao — hành vi này giữ nguyên y hệt bản gốc, không nằm trong phạm vi 3 bug đã duyệt sửa lần này.

Cả hai đều đã được ghi chú trực tiếp trong code/README để không bị quên khi có ai đó động vào khu vực này sau này.

## 8. Thay đổi cần biết khi cập nhật lên bản refactor

- **Bắt buộc mới**: `CHAT_BASE_URL`, `CHAT_API_KEY`, `CHAT_MODEL_NAME`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL_NAME` — ai đã có `.env` đầy đủ theo `.env.example` thì không bị ảnh hưởng.
- **Không đổi**: biến `PORT`, cách chạy `python main.py`, định dạng/tên file trong `data/`, request shape của `/chat`.
- **Đổi có chủ đích**: response `stream=false` giờ trả `sources` đầy đủ thay vì luôn rỗng; `/no_think` giờ thực sự được gửi lên LLM (trước đây tài liệu mô tả nhưng chưa chạy).
- File cũ đã xóa: `services/Chat.py`, `services/Search.py`, `services/RAGPipeline.py`, `services/OpenAIExtended.py` — thay bằng các module trong `core/`, `providers/`, `services/search.py`, `services/rag_pipeline.py`.
- Cài đặt: `pip install -r requirements.txt` (chạy API), thêm `-r requirements-pipeline.txt` nếu chạy `pipeline/`, thêm `-r requirements-dev.txt` nếu chạy test.
