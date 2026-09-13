# Direction: Deploying to AWS / GCP

> Infrastructure roadmap — not yet implemented. Based on research as of 2026-08 for this specific system: FastAPI + SSE streaming (runs the blocking pipeline in a separate thread, pushed through a `queue.Queue`), a ~3.2GB FAISS index, LLM/embedding/reranker via an OpenAI-compatible endpoint.

## 1. System-specific requirements to keep in mind when choosing infrastructure

- **Long-lived SSE streaming**: `api/app.py::chat_endpoint` keeps the connection open while the pipeline runs (sub-query → retrieval → tool-call loop → answer streaming) — the infrastructure must tolerate **long request/response times**, which rules out services with short hard timeouts.
- **3.2GB FAISS index** — needs to be loaded into RAM at startup (`FaissVectorStore.__init__` calls `faiss.read_index` synchronously, blocking) — affects cold-start time if using scale-to-zero compute.
- **LLM/embedding/reranker** are currently external services (self-hosted via llama.cpp/vLLM or LM Studio) — the infrastructure decision for this part is **independent** from the infrastructure for the API part (`core/config.py` already separates `CHAT_BASE_URL`/`EMBEDDING_BASE_URL`/`RERANK_BASE_URL` into their own config, so the two parts can live in different places without any code changes).
- Small team → prioritize the option that requires **less self-operation** over the option optimized for cost at large scale.

## 2. AWS vs GCP, layer by layer

### 2.1. Compute for the API layer (FastAPI + SSE)

| | AWS | GCP |
|---|---|---|
| Suitable choice | **ECS Fargate** + ALB | **Cloud Run** |
| Why | No request-processing time limit, works well behind an ALB for long-lived SSE connections, free control plane | Timeout can be raised up to 60 minutes (enough for most Q&A turns), supports SSE/WebSocket |
| Avoid | **AWS App Runner** — a hard 120-second timeout that kills SSE streams; a feature request for SSE support has been outstanding for years without AWS shipping it | — |
| When to move to K8s | EKS is only needed if you want the same cluster to also host GPU nodes for inference, or need deeper network/scaling control than Fargate allows | GKE Autopilot is only needed if you want connections with no absolute time limit, or want to combine API + inference in 1 cluster |

**Notable difference**: GCP Cloud Run now supports attaching **NVIDIA L4 GPUs** directly, with scale-to-zero and cold start down to a few seconds — AWS has no equivalent at the serverless layer (App Runner/Fargate don't support GPU at all).

### 2.2. Hosting LLM/Embedding/Reranker

| | AWS | GCP |
|---|---|---|
| Self-hosting on GPU | EC2 **G5** (A10G, cost-effective for moderate load) / **G6** (L40S, higher throughput) / P4d for large models | GCP **G2** (L4, sweet spot for 7-9B-class models) / A2 (A100) for large models/fine-tuning; **Cloud Run GPU** is a particularly good fit for a small team — run vLLM without having to manage a GPU cluster yourself |
| Managed inference | Amazon Bedrock (already has an OpenAI-style compatibility layer but doesn't cover every model/operation yet); SageMaker endpoint self-hosting a vLLM container | Vertex AI Model Garden (no built-in OpenAI-compatible endpoint — needs a shim like LiteLLM); Vertex custom endpoint similar to SageMaker |
| Recommendation | If self-hosting: EC2 G5/G6 running vLLM, exposing vLLM's built-in OpenAI-compatible endpoint (matches `core/config.py::ChatSettings/EmbeddingSettings` exactly, no shim needed) | If self-hosting: Cloud Run GPU is the lightest-ops option for a small team |

### 2.3. Storing the FAISS index (~3.2GB) + JSON metadata

- **Simplest & cheapest**: bake the index into the image or download it from **S3/GCS to local disk at startup** (baked into the container or an init container) — 3.2GB loads into RAM in a few seconds, no need for a shared filesystem since it's currently single-writer/read-mostly (exactly how `pipeline/chunk_embedding.py` generates the files once and `services/search.py` only reads them).
- **EFS/FSx (AWS) or Filestore (GCP)** is only worth investing in if multiple instances need to live-reload a frequently-updated index — currently the offline pipeline runs in batches, with no real-time update requirement, so this **isn't needed** at this stage.
- **Moving to a managed vector DB** (Amazon OpenSearch k-NN, Qdrant Cloud, Pinecone, Vertex AI Vector Search) is only truly necessary when: concurrent writes are needed, filter/hybrid search more complex than `FaissVectorStore` currently supports is needed, or scale exceeds an estimated ~50-100M vectors / >$500/month infrastructure. At the current scale (~810k chunks per the README), this is **far from** that threshold — sticking with FAISS + local disk is reasonable until data update frequency or complex filtering requirements increase.

### 2.4. Other supporting concerns

| | AWS | GCP |
|---|---|---|
| Secrets | Parameter Store (SecureString, free) is enough unless automatic rotation is needed (only then is Secrets Manager needed, which costs) | Secret Manager charges per call, which can add disproportionate cost at small scale |
| Observability | CloudWatch (metrics/logs) + X-Ray (tracing) — 2 separate services that need wiring together | Cloud Logging/Trace/Monitoring is integrated out of the box, more streamlined for a small team |
| Autoscaling | ECS Fargate target-tracking on CPU/request-count | Cloud Run scale-to-zero — fits uneven Q&A traffic |
| GPU inference scaling | Keep at least 1 "warm" instance to avoid a multi-minute cold start when loading a multi-GB model | Similar |

### 2.5. Cost comparison (relative, not exact pricing)

- **Self-hosting GPU vs. using a managed LLM API**: self-hosting only pays off when volume is large enough (break-even estimated at roughly a few million tokens/day or more) **and** operational/engineering cost is factored in (~$3,000-6,000/month in effort) — for a small team, a managed API is usually **cheaper in total cost** even though self-hosted price/token looks cheaper on paper.
- **AWS vs GCP overall**: GPU pricing is roughly equivalent between the two (G5/G6 ≈ G2/A2), both offer 60-91% spot discounts; the real difference is that **Cloud Run GPU significantly cuts operational cost** (no need to manage a GPU cluster yourself) compared to AWS, which only offers the EC2/EKS route for GPU.

## 3. Concrete recommended path

Given the team's current small size, here are 2 roughly equivalent options — pick based on the team's existing experience:

**GCP option** (least self-operated infrastructure):
```
Cloud Run (API, 60-minute timeout) + Cloud Run GPU or 1 G2/L4 VM running vLLM
    + GCS (source) → local disk at startup for FAISS
    + Secret Manager + Cloud Logging/Trace
```

**AWS option** (if the team is already familiar with the AWS ecosystem):
```
ECS Fargate + ALB (API) + 1 EC2 G5/G6 (or a 1-2 instance ASG) running vLLM
    + S3 (source) → local EBS at startup for FAISS
    + Parameter Store + CloudWatch/X-Ray
```

**Whichever is chosen**:
1. Start with a **managed LLM API** (instead of self-hosting GPU right away) unless volume is already large or there's a hard requirement that data not leave self-managed infrastructure (data residency — an important consideration for legal data). Since `core/config.py` already abstracts `CHAT_BASE_URL`/`EMBEDDING_BASE_URL` behind an interface, switching from self-host to managed API (or vice versa) later is just an environment variable change, no code changes needed.
2. Keep FAISS as a flat file loaded to local disk until there's a real need to migrate to a managed vector DB (see the threshold in section 2.3).
3. Containerizing the app (`Dockerfile` for `api/`, separate for the pipeline if batch runs are needed on Cloud Run Jobs / AWS Batch) is a shared preparation step, doable before committing to AWS or GCP.

## 4. Specific risks/pitfalls to avoid

- **Don't use AWS App Runner** for the API layer — the 120s timeout will cut off an SSE stream mid-flight.
- With Cloud Run, remember to **raise the timeout to its maximum (60 minutes)** and enable session affinity if the client reconnects the SSE stream.
- ECS Fargate **doesn't support GPU** — the GPU inference part (if self-hosted) must always be separated onto EC2/EKS, it can't be combined into the same task definition as the API.
- Don't load the FAISS index over a shared filesystem (EFS/Filestore) unless truly necessary — it adds unnecessary latency and cost at the current scale.

## References

- App Runner SSE timeout issue — https://repost.aws/questions/QUHFHBKsCYQlueDcXywdZ5jw/apprunner-timeout-for-sse-connection , https://github.com/aws/apprunner-roadmap/issues/23
- ECS Fargate for long-lived connections — https://github.com/aws/containers-roadmap/issues/88
- Cloud Run WebSocket/SSE timeout — https://docs.cloud.google.com/run/docs/triggering/websockets
- Cloud Run GPU (GA) — https://cloud.google.com/blog/products/serverless/cloud-run-gpus-are-now-generally-available
- EC2 instance types for LLM inference — https://markaicode.com/best/best-amazon-ec2-instance-types-llm-inference-production/
- Bedrock OpenAI-compatibility — https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html
- FAISS on AWS (reference architecture) — https://airbyte.com/blog/aws-ai-chatbot-using-faiss-vector-store
- Managed vector DB comparison 2026 — https://www.digitalapplied.com/blog/vector-databases-for-ai-agents-pinecone-qdrant-2026
- Self-host vs managed LLM API cost comparison — https://www.sitepoint.com/local-llms-vs-cloud-api-cost-analysis-2026/
