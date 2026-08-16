# Định hướng triển khai lên AWS / GCP

> Roadmap hạ tầng — chưa implement. Dựa trên research thời điểm 2026-08 cho hệ thống cụ thể: FastAPI + SSE streaming (chạy pipeline blocking trong thread riêng, đẩy qua `queue.Queue`), FAISS index ~3.2GB, LLM/embedding/reranker qua endpoint tương thích OpenAI.

## 1. Yêu cầu đặc thù của hệ thống cần lưu ý khi chọn hạ tầng

- **SSE streaming dài hơi**: `api/app.py::chat_endpoint` giữ kết nối mở trong lúc pipeline chạy (sub-query → retrieval → tool-call loop → answer streaming) — hạ tầng phải chấp nhận **request/response time dài**, không phù hợp với các dịch vụ có timeout cứng ngắn.
- **FAISS index 3.2GB** — cần load vào RAM khi khởi động (`FaissVectorStore.__init__` gọi `faiss.read_index` đồng bộ, blocking) — ảnh hưởng tới cold-start time nếu dùng compute scale-to-zero.
- **LLM/embedding/reranker** hiện là service ngoài (tự host qua llama.cpp/vLLM hoặc LM Studio) — quyết định hạ tầng cho phần này **độc lập** với hạ tầng cho phần API (`core/config.py` đã tách `CHAT_BASE_URL`/`EMBEDDING_BASE_URL`/`RERANK_BASE_URL` thành config riêng, nên 2 phần có thể nằm ở 2 nơi khác nhau mà không cần sửa code).
- Team nhỏ → ưu tiên phương án **ít phải tự vận hành** hơn là phương án tối ưu chi phí ở scale lớn.

## 2. So sánh AWS vs GCP theo từng lớp

### 2.1. Compute cho tầng API (FastAPI + SSE)

| | AWS | GCP |
|---|---|---|
| Lựa chọn phù hợp | **ECS Fargate** + ALB | **Cloud Run** |
| Vì sao | Không giới hạn thời gian xử lý request, chạy tốt sau ALB cho kết nối SSE dài, control plane miễn phí | Timeout có thể tăng tới 60 phút (đủ cho hầu hết lượt hỏi-đáp), hỗ trợ SSE/WebSocket |
| Cần tránh | **AWS App Runner** — timeout cứng 120 giây, giết chết SSE stream; đã có yêu cầu hỗ trợ SSE tồn đọng nhiều năm chưa được AWS triển khai | — |
| Khi nào cần lên K8s | EKS chỉ cần thiết nếu muốn cùng cluster host cả GPU node cho inference, hoặc cần kiểm soát network/scaling sâu hơn mức Fargate cho phép | GKE Autopilot chỉ cần nếu muốn kết nối không giới hạn thời gian tuyệt đối, hoặc gộp API + inference vào 1 cluster |

**Điểm khác biệt đáng chú ý**: GCP Cloud Run hiện hỗ trợ gắn thẳng **GPU NVIDIA L4**, scale-to-zero, cold start tới khi sẵn sàng chỉ vài giây — AWS chưa có tương đương ở tầng serverless (App Runner/Fargate hoàn toàn không hỗ trợ GPU).

### 2.2. Hosting LLM/Embedding/Reranker

| | AWS | GCP |
|---|---|---|
| Tự host GPU | EC2 **G5** (A10G, hiệu quả chi phí cho tải vừa) / **G6** (L40S, throughput cao hơn) / P4d cho model lớn | GCP **G2** (L4, điểm ngọt cho model cỡ 7-9B) / A2 (A100) cho model lớn/fine-tune; **Cloud Run GPU** là lựa chọn đặc biệt phù hợp team nhỏ — chạy vLLM mà không cần tự quản lý cluster GPU |
| Managed inference | Amazon Bedrock (đã có lớp tương thích OpenAI-style API nhưng chưa phủ hết mọi model/operation); SageMaker endpoint tự host container vLLM | Vertex AI Model Garden (không có endpoint tương thích OpenAI sẵn — cần shim như LiteLLM); Vertex custom endpoint tương tự SageMaker |
| Khuyến nghị | Nếu tự host: EC2 G5/G6 chạy vLLM, expose OpenAI-compatible endpoint sẵn có của vLLM (khớp đúng `core/config.py::ChatSettings/EmbeddingSettings` hiện tại, không cần shim) | Nếu tự host: Cloud Run GPU là lựa chọn vận hành nhẹ nhất cho team nhỏ |

### 2.3. Lưu trữ FAISS index (~3.2GB) + JSON metadata

- **Đơn giản & rẻ nhất**: đóng gói index vào image hoặc tải từ **S3/GCS về local disk lúc khởi động** (bake vào container hoặc init container) — 3.2GB load vào RAM trong vài giây, không cần shared filesystem vì hiện tại là single-writer/read-mostly (đúng như cách `pipeline/chunk_embedding.py` sinh ra file 1 lần, `services/search.py` chỉ đọc).
- **EFS/FSx (AWS) hoặc Filestore (GCP)** chỉ đáng đầu tư nếu có nhiều instance cần live-reload index thường xuyên cập nhật — hiện tại pipeline offline chạy theo lô (batch), không có yêu cầu real-time update, nên **chưa cần** ở giai đoạn này.
- **Chuyển sang managed vector DB** (Amazon OpenSearch k-NN, Qdrant Cloud, Pinecone, Vertex AI Vector Search) chỉ thực sự cần khi: cần ghi đồng thời (concurrent write), cần filter/hybrid search phức tạp hơn khả năng hiện tại của `FaissVectorStore`, hoặc quy mô vượt ngưỡng ước tính ~50-100M vector / >500 USD/tháng hạ tầng. Ở quy mô hiện tại (~810k chunks theo README) **còn rất xa** ngưỡng này — giữ nguyên FAISS + local disk là hợp lý cho tới khi tần suất cập nhật dữ liệu hoặc yêu cầu filter phức tạp tăng lên.

### 2.4. Các mối quan tâm hỗ trợ khác

| | AWS | GCP |
|---|---|---|
| Secrets | Parameter Store (SecureString, miễn phí) đủ dùng trừ khi cần rotation tự động (khi đó mới cần Secrets Manager, có phí) | Secret Manager tính phí theo lượt gọi, có thể tăng chi phí không cân xứng ở scale nhỏ |
| Observability | CloudWatch (metric/log) + X-Ray (trace) — 2 dịch vụ tách rời, cần tự nối | Cloud Logging/Trace/Monitoring tích hợp sẵn, gọn hơn cho team nhỏ |
| Autoscaling | ECS Fargate target-tracking theo CPU/request-count | Cloud Run scale-to-zero — phù hợp traffic hỏi-đáp không đều |
| GPU inference scaling | Giữ tối thiểu 1 instance "ấm" để tránh cold-start vài phút khi load model nhiều GB | Tương tự |

### 2.5. So sánh chi phí (tương đối, không phải giá chính xác)

- **Tự host GPU vs dùng managed LLM API**: tự host chỉ có lợi khi lưu lượng đủ lớn (ước tính hoà vốn ở khoảng vài triệu token/ngày trở lên) **và** đã tính cả chi phí vận hành/kỹ sư (~3.000-6.000 USD/tháng công sức) — với team nhỏ, dùng managed API thường **rẻ hơn về tổng chi phí** dù giá/token tự host trông rẻ hơn trên giấy.
- **AWS vs GCP nói chung**: giá GPU tương đương giữa 2 bên (G5/G6 ≈ G2/A2), cả 2 đều có giảm giá spot 60-91%; điểm khác biệt thực sự là **Cloud Run GPU giảm đáng kể chi phí vận hành** (không cần tự quản lý cluster GPU) so với việc AWS chỉ có đường EC2/EKS cho GPU.

## 3. Khuyến nghị lộ trình cụ thể

Với quy mô team nhỏ hiện tại, gợi ý 2 phương án tương đương, chọn theo kinh nghiệm sẵn có của team:

**Phương án GCP** (ít hạ tầng phải tự vận hành nhất):
```
Cloud Run (API, timeout 60 phút) + Cloud Run GPU hoặc 1 VM G2/L4 chạy vLLM
    + GCS (nguồn) → local disk lúc khởi động cho FAISS
    + Secret Manager + Cloud Logging/Trace
```

**Phương án AWS** (nếu team đã quen hệ sinh thái AWS):
```
ECS Fargate + ALB (API) + 1 EC2 G5/G6 (hoặc ASG 1-2 instance) chạy vLLM
    + S3 (nguồn) → local EBS lúc khởi động cho FAISS
    + Parameter Store + CloudWatch/X-Ray
```

**Dù chọn bên nào**:
1. Bắt đầu bằng **managed LLM API** (thay vì tự host GPU ngay) trừ khi đã có lưu lượng lớn hoặc yêu cầu bắt buộc dữ liệu không rời khỏi hạ tầng tự quản (data residency — cân nhắc quan trọng với dữ liệu pháp lý). Vì `core/config.py` đã trừu tượng hoá `CHAT_BASE_URL`/`EMBEDDING_BASE_URL` qua interface, việc đổi từ self-host sang managed API (hoặc ngược lại) sau này chỉ là đổi biến môi trường, không cần sửa code.
2. Giữ FAISS ở dạng file phẳng tải vào local disk cho tới khi thật sự cần migrate sang managed vector DB (xem ngưỡng ở mục 2.3).
3. Container hoá app (`Dockerfile` cho `api/`, tách riêng cho pipeline nếu cần chạy batch trên Cloud Run Jobs / AWS Batch) là bước chuẩn bị chung, làm được trước khi chốt chọn AWS hay GCP.

## 4. Rủi ro/pitfall cụ thể cần tránh

- **Không dùng AWS App Runner** cho tầng API — timeout 120s sẽ cắt ngang SSE stream giữa chừng.
- Với Cloud Run, nhớ **tăng timeout lên mức tối đa (60 phút)** và bật session affinity nếu client reconnect SSE.
- ECS Fargate **không hỗ trợ GPU** — phần inference GPU (nếu tự host) luôn phải tách riêng sang EC2/EKS, không thể gộp chung task definition với API.
- Đừng tải FAISS index qua shared filesystem (EFS/Filestore) nếu chưa thực sự cần — thêm độ trễ và chi phí không cần thiết ở quy mô hiện tại.

## Nguồn tham khảo

- App Runner SSE timeout issue — https://repost.aws/questions/QUHFHBKsCYQlueDcXywdZ5jw/apprunner-timeout-for-sse-connection , https://github.com/aws/apprunner-roadmap/issues/23
- ECS Fargate cho long-lived connection — https://github.com/aws/containers-roadmap/issues/88
- Cloud Run WebSocket/SSE timeout — https://docs.cloud.google.com/run/docs/triggering/websockets
- Cloud Run GPU (GA) — https://cloud.google.com/blog/products/serverless/cloud-run-gpus-are-now-generally-available
- EC2 instance types cho LLM inference — https://markaicode.com/best/best-amazon-ec2-instance-types-llm-inference-production/
- Bedrock OpenAI-compatibility — https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html
- FAISS trên AWS (kiến trúc tham khảo) — https://airbyte.com/blog/aws-ai-chatbot-using-faiss-vector-store
- So sánh vector DB managed 2026 — https://www.digitalapplied.com/blog/vector-databases-for-ai-agents-pinecone-qdrant-2026
- So sánh chi phí self-host vs managed LLM API — https://www.sitepoint.com/local-llms-vs-cloud-api-cost-analysis-2026/
