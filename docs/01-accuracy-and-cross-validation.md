# Định hướng: Tăng độ chính xác tối đa & Cross-validation đa nguồn

> Roadmap kiến trúc — chưa implement. Căn cứ trên khảo sát thị trường ở [02-hallucination-mitigation-landscape.md](02-hallucination-mitigation-landscape.md).

## 1. Mục tiêu

- Tăng độ chính xác của câu trả lời lên mức tối đa có thể trong giới hạn hạ tầng hiện tại (FAISS + LLM tương thích OpenAI).
- Giảm hallucination bằng cách **xác minh** (verify) câu trả lời trước khi trả về, thay vì chỉ tin tưởng tuyệt đối vào 1 lần sinh của LLM.
- Cross-validate với nhiều nguồn (nhiều lượt retrieval, nhiều mô hình, và cả web search) khi độ tin cậy của nguồn nội bộ (FAISS) thấp.
- Giữ nguyên nguyên tắc kiến trúc interface-first đã thiết lập (xem `REPORT.md`) — mọi thành phần mới đều là 1 interface trong `core/interfaces.py` + 1 cài đặt cụ thể trong `providers/`, không sửa `services/rag_pipeline.py` theo kiểu hardcode if/else.

## 2. Điểm hallucination có thể xảy ra trong pipeline hiện tại

Nhìn lại `services/rag_pipeline.py`, hallucination có thể phát sinh ở 3 chỗ:

1. **Sub-query decomposition** — LLM tự tách câu hỏi, có thể tách sai/thiếu ý → context retrieval bị lệch ngay từ đầu.
2. **Retrieval** — FAISS trả về chunk không thực sự liên quan (điểm rerank cao nhưng nội dung không khớp), LLM vẫn phải tổng hợp từ context đó.
3. **Answer generation** — LLM có thể diễn giải sai nội dung chunk, gắn nhầm số trích dẫn `[N]`, hoặc "bịa" thêm chi tiết không có trong context dù đã được nhắc citation rule trong prompt.

Hiện tại **không có bước nào kiểm tra lại** câu trả lời sau khi sinh ra — đây là khoảng trống lớn nhất cần lấp theo khảo sát ở tài liệu 02 (kỹ thuật "citation/groundedness verification" là kỹ thuật có ROI cao nhất và dễ bolt-on nhất).

## 3. Kiến trúc đề xuất

### 3.1. Interface mới (`core/interfaces.py`)

```python
class VerificationResult(TypedDict):
    claim: str
    supported: bool          # có được nguồn trích dẫn hỗ trợ không
    confidence: float        # 0..1
    reason: str               # lý do ngắn gọn (để log/debug)

class Verifier(Protocol):
    def verify(self, claim: str, source_text: str) -> VerificationResult: ...

class WebSearchProvider(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Trả về [{title, url, snippet}], dùng để cross-check khi FAISS
        không đủ tin cậy hoặc văn bản có thể đã hết hiệu lực."""
        ...

class ConfidenceScorer(Protocol):
    def score(self, *, retrieval_scores: list[float], verification_results: list[VerificationResult]) -> float:
        """Gộp điểm retrieval + điểm verify thành 1 confidence score duy nhất,
        dùng để quyết định: trả lời thẳng / bổ sung web search / abstain."""
        ...
```

### 3.2. Cài đặt cụ thể (`providers/`)

| Provider | Vai trò | Ghi chú |
|---|---|---|
| `providers/quote_verifier.py::VerbatimQuoteVerifier` | Kiểm tra câu trích dẫn có khớp verbatim (substring/fuzzy) với `content` của chunk được cite hay không | Thuần Python, không cần model, **chi phí gần như 0** — nên làm đầu tiên |
| `providers/hhem_verifier.py::HHEMVerifier` | Dùng Vectara HHEM (cross-encoder mã nguồn mở, tự host) chấm điểm entailment giữa câu trả lời và chunk nguồn | Nhanh hơn nhiều so với dùng LLM-judge (theo khảo sát 02); có thể chạy trên CPU cho model nhỏ |
| `providers/llm_judge_verifier.py::LLMJudgeVerifier` | Dùng chính `LLMProvider` hiện có, hỏi thêm 1 lần "câu trả lời này có được hỗ trợ bởi đoạn văn bản sau không?" | Không cần thêm hạ tầng, nhưng tốn thêm 1 lượt gọi LLM/claim — nên dùng cho sampled review, không phải mọi request |
| `providers/tavily_search.py::TavilySearchProvider` (hoặc Exa) | Web search khi cần cross-check | Chỉ gọi khi confidence thấp, không gọi mặc định (tránh tăng latency/cost cho mọi câu hỏi) |
| `providers/weighted_confidence_scorer.py::WeightedConfidenceScorer` | Công thức đơn giản: `confidence = w1*avg(retrieval_score) + w2*avg(verification.confidence)` | Bắt đầu bằng công thức tuyến tính đơn giản, tinh chỉnh trọng số qua tập eval (xem mục 5) |

### 3.3. Bước xử lý mới trong `RAGPipeline.process()`

Thêm 1 bước **verification** sau khi có `full_answer`, trước khi phát event `answer/done`:

```
[Bước 3+4] Answer (như hiện tại)
    │
    ▼
[Bước 5 — MỚI] Verification
    └─► Parse các câu trích dẫn [N] trong full_answer
    └─► Với mỗi câu có citation: verify(claim, content_của_chunk_N)
        (chạy VerbatimQuoteVerifier trước — rẻ; nếu nghi ngờ mới chạy HHEMVerifier)
    └─► ConfidenceScorer.score(...) → confidence tổng
    └─► Nếu confidence thấp:
          - Thử bổ sung nguồn qua WebSearchProvider (nếu bật)
          - Hoặc gắn cờ "cần xác minh thêm" vào response thay vì trả lời chắc chắn
          - Hoặc abstain hẳn ("Không đủ căn cứ để trả lời chính xác câu hỏi này")
    │
    ▼
answer/done (kèm thêm field mới: "confidence", "verification": [...])
```

Event SSE mới đề xuất thêm vào `core/models.py::EventStep`: `VERIFICATION = "verification"`, phát ra dạng `{"step": "verification", "status": "done", "data": {"confidence": 0.82, "flags": [...]}}` — tuân theo đúng pattern event hiện có, frontend có thể chọn hiển thị hoặc bỏ qua field mới mà không vỡ luồng cũ.

### 3.4. Cải thiện retrieval (không cần chờ verification)

Có thể làm song song, độc lập với bước verification:

- **RAG-Fusion cho sub-query**: bước sub-query hiện tại đã tách câu hỏi thành nhiều sub-query rồi gộp kết quả bằng dedupe theo `chunk_id` (`_deduplicate_docs`). Nâng cấp lên **Reciprocal Rank Fusion** (RRF) thay vì dedupe đơn giản — với mỗi chunk xuất hiện ở nhiều sub-query, cộng dồn điểm theo rank thay vì chỉ giữ bản đầu tiên gặp. Đây là thay đổi nhỏ trong `SemanticSearchService`/`RAGPipeline`, không cần thêm interface mới.
- **Self-consistency có chọn lọc**: với câu hỏi được đánh dấu "high-stakes" (vd. do người dùng gắn cờ, hoặc confidence thấp ở vòng đầu), chạy lại bước answer generation N=3 lần (temperature > 0), so khớp câu trả lời — nếu đồng nhất thì tăng confidence, nếu phân kỳ thì hạ confidence/abstain. Chi phí N× lượt gọi LLM nên **chỉ bật có điều kiện**, không bật mặc định cho mọi câu hỏi.

## 4. Lộ trình triển khai theo giai đoạn

| Giai đoạn | Việc làm | Chi phí/rủi ro | Vì sao làm trước/sau |
|---|---|---|---|
| **Phase 1** | `VerbatimQuoteVerifier` — kiểm tra citation `[N]` có tồn tại trong `citation_map` + nội dung trích có khớp verbatim/fuzzy với chunk | Gần như 0 (thuần Python, không gọi thêm API) | ROI cao nhất, rẻ nhất, bắt được lỗi "bịa trích dẫn" — nên làm ngay |
| **Phase 2** | `HHEMVerifier` (self-host model nhỏ) + `WeightedConfidenceScorer` + ngưỡng abstain | Trung bình — cần tự host 1 model nhỏ, thêm 1 bước tính điểm mỗi request | Cần Phase 1 xong để có dữ liệu citation sạch làm input cho verifier |
| **Phase 3** | `WebSearchProvider` (Tavily/Exa) — chỉ gọi khi confidence từ Phase 2 thấp | Trung bình — thêm chi phí API bên ngoài, cần xử lý rate limit/timeout | Chỉ có ý nghĩa sau khi đã đo được "khi nào confidence thấp" ở Phase 2 |
| **Phase 4 (tuỳ chọn)** | Self-consistency/ensemble cho câu hỏi high-stakes | Cao nhất (N× LLM call) | Chỉ nên bật có điều kiện, sau khi đã có cơ chế đánh dấu "high-stakes" từ Phase 2/3 |

## 5. Đo lường hiệu quả

Trước khi tối ưu, cần 1 tập câu hỏi eval cố định (có thể lấy từ chính leaderboard R2AI 2026 hoặc tự tạo) để đo:

- **Faithfulness/groundedness** qua RAGAS hoặc DeepEval (chạy offline trong CI, không phải runtime) — xem tài liệu 02 mục "Evaluation Frameworks".
- Theo dõi số lượng câu trả lời bị abstain / gắn cờ "cần xác minh" theo thời gian — tỉ lệ abstain quá cao nghĩa là retrieval hoặc threshold cần điều chỉnh, không phải verifier bị lỗi.

## 6. Đánh đổi cần lưu ý

- Mỗi bước verification/cross-validation đều **tăng latency** — với 1 pipeline đã có tới `MAX_TOOL_ITERATIONS=3` vòng lặp tool-call, cần cân nhắc verification chạy song song (async) với việc stream câu trả lời ra client thay vì chặn cứng, để không làm chậm trải nghiệm streaming đã có.
- Web search (Phase 3) đưa vào 1 nguồn dữ liệu **không kiểm soát được** (không qua pipeline chunk/embed nội bộ) — cần rõ ràng với người dùng khi câu trả lời có phần dựa vào web search thay vì chỉ dựa vào kho văn bản pháp luật đã qua xử lý, tương tự cách hiển thị `sources` hiện tại nhưng phân biệt rõ nguồn nội bộ vs nguồn web.
