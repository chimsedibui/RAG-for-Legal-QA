"""Shared data shapes used across the offline pipeline, services, and API.

EventStep/EventStatus give the SSE event dicts yielded by RAGPipeline.process()
a typed vocabulary instead of ad hoc string literals scattered across
services/rag_pipeline.py and api/app.py.

Chunk/ChunkMetadata mirror the on-disk chunk_map.json/chunks.json schema
produced by pipeline/chunk_embedding.py and consumed by services/search.py, so
both sides share one definition of the field names instead of duplicating
string literals independently.
"""
from enum import Enum
from typing import Optional, TypedDict


class EventStep(str, Enum):
    SUB_QUERIES = "sub_queries"
    RETRIEVAL = "retrieval"
    CONTEXT_READY = "context_ready"
    TOOL_CALL = "tool_call"
    ANSWER = "answer"


class EventStatus(str, Enum):
    PROCESSING = "processing"
    DONE = "done"
    START = "start"
    STREAMING = "streaming"
    DETECTED = "detected"
    EXECUTED = "executed"
    ERROR = "error"


class ChunkMetadata(TypedDict, total=False):
    title: str
    doc_num: str
    doc_id: str
    article: str
    clause: str


class Chunk(TypedDict):
    chunk_id: str
    embed_text: str
    metadata: ChunkMetadata
