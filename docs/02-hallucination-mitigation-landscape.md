# Khảo sát: Các phương pháp chống Hallucination hiện có trên thị trường (2025-2026)

> Tài liệu khảo sát (research), làm căn cứ cho định hướng kiến trúc ở [01-accuracy-and-cross-validation.md](01-accuracy-and-cross-validation.md). Nguồn tổng hợp qua research thời điểm 2026-08.

## 1. Vì sao cần quan tâm đặc biệt tới hallucination trong lĩnh vực pháp lý

Nghiên cứu của Stanford RegLab/HAI (Magesh et al., *Journal of Empirical Legal Studies* 2025) kiểm thử hơn 200 câu hỏi pháp lý trên các sản phẩm AI pháp lý thương mại:

- **Lexis+ AI**: hallucinate >17% số lần.
- **Westlaw AI-Assisted Research**: hallucinate >34% số lần.
- LexisNexis sau đó phải rút lại tuyên bố marketing "100% không hallucination", giới hạn lại chỉ còn áp dụng cho phần "linked citations".

Theo một tracker công khai, tính tới giữa 2026 đã có **hơn 1.590 vụ việc được ghi nhận** có trích dẫn do AI "bịa" xuất hiện trong hồ sơ tòa án trên toàn thế giới; riêng Q1/2026 tổng mức phạt vì lạm dụng AI trong tố tụng đã lên tới 145.000 USD. ABA Formal Opinion 512 (Mỹ) khẳng định luật sư chịu trách nhiệm hoàn toàn về output của AI bất kể dùng công cụ nào.

→ Kết luận: hallucination trong lĩnh vực pháp lý **không phải rủi ro lý thuyết**, các sản phẩm thương mại lớn hiện tại vẫn hallucinate ở mức 2 chữ số phần trăm dù đã đầu tư mạnh — nên bất kỳ cải thiện nào cũng cần đo lường được (xem mục 5 tài liệu 01), không chỉ dựa vào cảm giác "có vẻ tốt hơn".

## 2. Các kỹ thuật kiến trúc (architectural patterns)

| Kỹ thuật | Cách hoạt động | Dễ/khó bolt-on vào pipeline hiện tại |
|---|---|---|
| **Corrective RAG (CRAG)** | 1 model nhỏ chấm điểm chunk retrieval là đúng/mơ hồ/sai; nếu sai → re-retrieve hoặc fallback sang web search; nếu mơ hồ → lọc bớt phần không liên quan trước khi đưa vào context | **Dễ** — chỉ là 1 bước scoring + rẽ nhánh, không cần sửa retriever hiện có. Báo cáo giảm ~30% hallucination trong thực tế |
| **Self-RAG** | Model tự sinh "reflection token" quyết định có cần retrieve tiếp không, và tự đánh giá câu trả lời có được context hỗ trợ không | **Khó hơn** — cần model được fine-tune/prompt riêng cho việc này; có thể mô phỏng bằng prompt "tự phê bình" nhưng kém tin cậy hơn bản gốc |
| **FLARE / DRAGIN** | Retrieve tiếp ngay giữa lúc đang sinh câu trả lời khi model "không chắc" về đoạn tiếp theo (dựa vào logprob) | **Trung bình** — cần LLM API trả về logprob, không phải endpoint OpenAI-compatible nào cũng hỗ trợ |
| **RAG-Fusion** | Sinh nhiều biến thể câu hỏi, retrieve riêng từng biến thể, gộp kết quả bằng Reciprocal Rank Fusion | **Rất dễ** — dự án đã có bước sub-query decomposition, chỉ cần đổi cách gộp kết quả (RRF thay vì dedupe đơn giản) |
| **Self-consistency / ensemble voting** | Chạy lại cùng 1 câu hỏi N lần (temperature > 0) hoặc qua nhiều model, biểu quyết theo đa số; câu trả lời càng đồng nhất thì càng đáng tin | **Dễ về kỹ thuật, tốn kém về chi phí** — chỉ nên bật có điều kiện cho câu hỏi quan trọng |
| **Citation/groundedness verification (NLI-based)** | Sau khi sinh câu trả lời, kiểm tra từng câu/claim có được đoạn văn bản trích dẫn "entail" (suy ra được) hay không, dùng model NLI/cross-encoder chuyên dụng | **Dễ, ROI cao nhất** — đây là kỹ thuật đơn lẻ đáng làm nhất theo khảo sát này |
| **Generate-Verify-Correct với verbatim quote** | Bắt LLM trích dẫn nguyên văn; script kiểm tra đoạn trích có thực sự tồn tại trong chunk nguồn (substring/fuzzy match) | **Rất dễ** — thuần Python, không cần model, độ chính xác cao cho văn bản pháp luật (vốn có câu chữ chính xác, ít đồng nghĩa) |
| **Confidence scoring + abstention** | Gộp điểm retrieval + điểm verify + độ đồng thuận self-consistency thành 1 confidence; dưới ngưỡng → từ chối trả lời/escalate người | **Dễ** — chỉ là công thức gộp điểm, không cần hạ tầng mới |
| **Human-in-the-loop** | Coi output AI là bản nháp; bắt buộc luật sư review trước khi dùng thật; không chấp nhận dùng 1 AI để tự verify AI khác thay cho review của con người | **Không phải hạ tầng — là quy trình/UX**: gắn cờ câu trả lời confidence thấp để bắt buộc review, tương tự cảnh báo y tế "tham khảo ý kiến chuyên gia" |

## 3. Công cụ đánh giá (evaluation frameworks) đang được dùng thực tế

| Công cụ | Loại | Dùng khi nào |
|---|---|---|
| **RAGAS** | Mã nguồn mở, tính faithfulness/context-precision/context-recall | Chạy offline trong CI trên tập eval cố định, không chạy real-time |
| **DeepEval** | Tương tự RAGAS, style pytest assertion | Thích hợp gate trong pipeline CI (fail build nếu faithfulness dưới ngưỡng) |
| **TruLens** | Tracing/feedback function cho production, hay đi kèm Langfuse | Quan sát chất lượng theo thời gian thực ở production |
| **Vectara HHEM (2.1)** | Cross-encoder mã nguồn mở, chuyên chấm điểm groundedness | Có thể tự host, nhanh hơn nhiều so với dùng LLM-judge (báo cáo: ~10 phút vs ~8 giờ cho cùng khối lượng đánh giá), độ chính xác benchmark ~78.9% — **ứng viên chính cho `HHEMVerifier` ở tài liệu 01** |
| **Patronus Lynx / Galileo Luna** | Model nhỏ chuyên phát hiện hallucination, nhanh/rẻ hơn LLM-as-judge | Nếu muốn dùng SaaS thay vì tự host |
| **Anthropic Citations API / Gemini Grounding with Search** | Tính năng grounding có sẵn của nhà cung cấp LLM | Chỉ dùng được nếu đổi sang Claude/Gemini; với endpoint OpenAI-compatible tổng quát vẫn cần tự làm bước verify riêng |

**Khuyến nghị stack cho dự án**: chạy 1 detector nhanh (HHEM) trên mọi/đa số request + LLM-as-judge (dùng `LLMProvider` sẵn có) trên tập mẫu để review sâu hơn định kỳ — không cần chờ tích hợp SaaS (Galileo/Patronus) ngay từ đầu.

## 4. Đặc thù pháp lý — grounding theo statute/case-law

- **Citation Grounding metric**: tỉ lệ % trích dẫn trong câu trả lời thực sự tồn tại (khớp với 1 node thật trong đồ thị/tập văn bản gốc) — dự án hiện đã có sẵn cấu trúc gần giống (`chunk_map.json`, `article_index_map.json`) nên việc thêm bước kiểm tra "citation `[N]` có ánh xạ đúng tới 1 chunk thật + nội dung trích khớp verbatim" gần như không tốn thêm hạ tầng.
- **Legal citation graph / Graph-RAG**: mô hình hoá quan hệ trích dẫn giữa văn bản-văn bản, điều-điều dưới dạng đồ thị (thường dùng Neo4j), vừa dùng để retrieval vừa dùng để validate câu trả lời. Đây là hướng đầu tư lớn hơn (tuần thay vì ngày) — chỉ nên cân nhắc nếu cần mô hình hoá quan hệ hiệu lực/sửa đổi văn bản pháp luật phức tạp (rất phù hợp bối cảnh luật Việt Nam hay có Nghị định/Thông tư sửa đổi, bổ sung, thay thế lẫn nhau).
- Với quy mô hiện tại của dự án (chunk theo Điều/Khoản/Điểm, không phải toàn văn bản), 1 phiên bản **rút gọn** của ý tưởng graph — bảng tra `doc_id + article → canonical text` (đã có sẵn dưới dạng `article_index_map.json`) — là điểm khởi đầu hợp lý, không cần dựng graph DB ngay.

## 5. Web-search-augmented RAG — cross-validate với nguồn ngoài

- **Pattern phổ biến**: query FAISS trước; nếu bước CRAG-style scoring hoặc bước verify sau khi sinh câu trả lời cho confidence thấp → fallback sang web search để đối chiếu, hoặc gắn cờ "cần xác minh thêm" thay vì im lặng trả lời sai.
- **Nhà cung cấp search API phổ biến**: Tavily (được coi là "tiêu chuẩn thực tế" cho AI agent năm 2025-2026, dễ tích hợp), Exa (semantic search), Bing Search API, SerpAPI, You.com API — khác nhau chủ yếu ở cách cấu trúc kết quả trả về và chi phí, không khác biệt lớn về chất lượng cho use-case cross-check.
- **Lưu ý riêng cho pháp luật Việt Nam**: web search có thể hữu ích để phát hiện văn bản đã **hết hiệu lực/được sửa đổi** mà kho dữ liệu nội bộ (crawl 1 lần, xem `pipeline/crawl_preprocess.py`) chưa cập nhật — đây là rủi ro hallucination đặc thù của domain pháp lý (thông tin đúng tại thời điểm crawl nhưng sai tại thời điểm hỏi), khác với hallucination "bịa nội dung" thông thường.

## 6. Khuyến nghị áp dụng cho dự án (xếp theo độ ưu tiên)

1. **Verbatim quote / citation existence check** — làm ngay, gần như miễn phí, bắt được lỗi trích dẫn sai/bịa.
2. **RAGAS/DeepEval trong CI** trên 1 tập câu hỏi eval cố định — để có con số đo lường trước khi tối ưu tiếp, tránh tối ưu "cảm tính".
3. **HHEM (hoặc tương đương) làm groundedness scorer runtime** — chi phí thấp, hiệu quả cao theo benchmark.
4. **RAG-Fusion cho sub-query** — cải thiện retrieval mà không cần thêm interface mới.
5. **Web search fallback có điều kiện** (Tavily/Exa) — chỉ khi độ tin cậy nội bộ thấp, đặc biệt hữu ích để bắt văn bản đã hết hiệu lực.
6. **Self-consistency/ensemble** — để cuối vì tốn kém, chỉ bật cho câu hỏi được đánh dấu high-stakes.

Chi tiết triển khai theo giai đoạn: xem [01-accuracy-and-cross-validation.md](01-accuracy-and-cross-validation.md) mục 4.

## Nguồn tham khảo

- Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools", *Journal of Empirical Legal Studies* (2025) — https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413
- CRAG (Corrective RAG) — https://github.com/HuskyInSalt/CRAG , https://openreview.net/forum?id=JnWJbrnaUE
- Self-consistency / LLM fan-out patterns — https://arxiv.org/pdf/2505.09031
- Auto-GDA (NLI-based groundedness) — https://arxiv.org/pdf/2410.03461
- Vectara HHEM benchmarking — https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking/
- DeepEval vs RAGAS vs TruLens — https://particula.tech/blog/deepeval-vs-ragas-vs-trulens-rag-evaluation-stack
- Anthropic Citations API — https://claude.com/blog/introducing-citations-api
- Gemini Grounding with Search — https://ai.google.dev/gemini-api/docs/google-search
- Legal citation graphs / Citation Grounding metric — https://arxiv.org/pdf/2606.00898 , https://arxiv.org/pdf/2605.28120
- Ontology-driven Graph RAG cho văn bản pháp luật — https://journals.sagepub.com/doi/10.3233/FAIA251598
- So sánh search API cho AI agent (Tavily/Exa/...) — https://brave.com/learn/best-search-api-2026/
