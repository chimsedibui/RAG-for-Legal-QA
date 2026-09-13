# Running the API with Docker Compose

Compose runs FastAPI; chat and embedding call OpenAI over the Internet. No GPU or vector database service needed. Reranker is disabled by default.

1. Create `.env` from `.env.example`, fill in both API keys (can be the same key). This machine is already configured from `api_key.txt`. Do not commit these files.
2. Prepare `data/faiss.index`, `data/faiss_id_map.json`, `data/chunk_map.json`, `data/article_index_map.json`, `data/chunks.json` from the same index-build run.
3. Run from the repo root:

```bash
docker compose config --quiet
docker compose up -d --build
docker compose logs -f api
curl http://localhost:8000/health
```

Web chat: http://localhost:8000 ; Swagger: http://localhost:8000/docs.
`/health` only checks the API, it doesn't call the model. Try a chat message to check the whole pipeline.

```bash
curl http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Bộ luật dân sự quy định những nguyên tắc cơ bản nào?"}],"stream":false}'
```

## If you don't have an index yet

You need `processed_data.json` in the output format of `pipeline/crawl_preprocess.py`, at the repo root. Once you have the data:

```bash
uv sync
uv run python pipeline/chunk_embedding.py
```

This command calls the embedding API and incurs cost based on the data. The sample config uses `text-embedding-3-small` with a default dimension of 1536, so `EMBEDDING_DIM=1536`. This variable sets the dimension of the new index, it does not change the dimension returned by the API. See [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings).

Don't reuse an index built by a different model even with the same dimension. When switching models, pick a new `DATA_DIR` and rebuild the whole index/maps from scratch; don't resume the old index. Compose mounts `DATA_DIR` from the host machine into `/app/data` read-only and doesn't create the directory automatically if it's missing.

Chat defaults to [gpt-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini). You can swap `CHAT_MODEL_NAME` for any model your account supports that's compatible with tool calling/JSON. A correctly formatted key doesn't prove the key is still valid or that the account has quota.

Change the exposed port with `PORT` in `.env`. Stop with `docker compose down`; host data is preserved.
