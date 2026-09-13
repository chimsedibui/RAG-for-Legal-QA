from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Literal
import json
import os
import queue
import threading
import asyncio

from core.config import get_settings
from providers.faiss_store import FaissVectorStore
from providers.openai_embedding import OpenAIEmbeddingProvider
from providers.openai_llm import OpenAILLMProvider
from providers.reranker import NullReranker, VLLMReranker
from services.rag_pipeline import RAGPipeline
from services.search import DocRefSearchService, SemanticSearchService, load_search_data
from tools.doc_ref_tool import SearchReferencedDocumentTool
from tools.registry import ToolRegistry

app = FastAPI(title="Legal RAG API")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


def build_pipeline() -> RAGPipeline:
    """Composition root: wires concrete providers/services/tools into a
    RAGPipeline. This is the one place that needs to change to swap a
    provider (e.g. a different LLM backend or vector store) or register a
    new tool — services/rag_pipeline.py itself only depends on interfaces."""
    settings = get_settings()

    llm = OpenAILLMProvider(settings.chat.base_url, settings.chat.api_key, settings.chat.model_name)
    embedder = OpenAIEmbeddingProvider(
        settings.embedding.base_url, settings.embedding.api_key, settings.embedding.model_name
    )
    reranker = (
        VLLMReranker(settings.rerank.base_url, settings.rerank.api_key, settings.rerank.model_name)
        if settings.reranker_enabled
        else NullReranker()
    )
    vector_store = FaissVectorStore(settings.data.faiss_index_path, settings.data.faiss_id_map_path)
    chunk_map, article_index_map, chunks_text_map = load_search_data(settings.data)

    semantic_search = SemanticSearchService(
        embedder, vector_store, reranker, chunk_map, chunks_text_map, settings.retrieval.threshold
    )
    doc_ref_search = DocRefSearchService(
        embedder, vector_store, chunk_map, article_index_map, chunks_text_map, settings.retrieval.threshold
    )

    tool_registry = ToolRegistry()
    tool_registry.register(
        SearchReferencedDocumentTool(doc_ref_search, top_k=settings.retrieval.tool_search_top_k)
    )

    return RAGPipeline(llm, semantic_search, tool_registry, settings.retrieval)


pipeline = build_pipeline()

# Sentinel signaling that the generator (running in a separate thread) has finished
_SENTINEL = object()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    allow_reasoning: bool = False


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


def _run_pipeline_in_thread(
    conversation: List[dict], stream: bool, allow_reasoning: bool, out_queue: "queue.Queue"
):
    """Run pipeline.process() (sync, blocking) in a separate thread.

    Since pipeline.process() is a sync generator containing blocking HTTP
    calls (LLM sub-query, semantic search, LLM streaming), running it directly
    with `for event in pipeline.process(...)` inside an `async def` would let
    each blocking step hold Uvicorn's event loop hostage, so events already
    yielded wouldn't get flushed to the socket right away -> the client would
    see "jerky" delivery, only getting data in bursts whenever some other
    async code happened to yield the CPU.

    Running the whole generator in a separate thread and pushing each event
    through a queue.Queue (thread-safe) lets every event get sent out as soon
    as it's ready, regardless of whether that thread is currently blocked on
    an HTTP call.
    """
    try:
        for event in pipeline.process(messages=conversation, stream=stream, allow_reasoning=allow_reasoning):
            out_queue.put(event)
    except Exception as e:
        out_queue.put({"step": "answer", "status": "error", "data": {"error": str(e)}})
    finally:
        out_queue.put(_SENTINEL)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "messages không được để trống."})

    conversation = [m.model_dump() for m in req.messages]

    if req.stream:
        async def event_generator():
            out_queue: "queue.Queue" = queue.Queue()
            thread = threading.Thread(
                target=_run_pipeline_in_thread,
                args=(conversation, True, req.allow_reasoning, out_queue),
                daemon=True,
            )
            thread.start()

            loop = asyncio.get_event_loop()

            try:
                while True:
                    event = await loop.run_in_executor(None, out_queue.get)

                    if event is _SENTINEL:
                        break

                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'step': 'answer', 'status': 'error', 'data': {'error': str(e)}}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                # Disable buffering on the reverse proxy side (e.g. nginx) so
                # SSE isn't held back in batches before reaching the browser.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-stream: run in a threadpool so the main event loop isn't blocked,
        # letting the server keep serving other requests concurrently.
        full_response = {
            "steps": [],
            "final_answer": "",
            "citations": {},
            "sources": []
        }

        def _run_non_stream():
            return list(pipeline.process(messages=conversation, stream=False, allow_reasoning=req.allow_reasoning))

        try:
            loop = asyncio.get_event_loop()
            steps = await loop.run_in_executor(None, _run_non_stream)

            full_response["steps"] = steps
            for event in steps:
                if event["step"] == "answer" and event["status"] == "done":
                    full_response["final_answer"] = event["data"].get("text", "")
                    full_response["citations"] = event["data"].get("citations", {})
                    full_response["sources"] = event["data"].get("sources", [])

            return JSONResponse(content=full_response)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_check():
    return {"status": "ok"}
