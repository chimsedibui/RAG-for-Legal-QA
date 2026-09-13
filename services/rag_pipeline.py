"""RAG orchestration: sub-query decomposition -> retrieval -> streaming
answer + tool-call loop. Depends only on the LLMProvider interface and the
services/registries it's handed at construction time — no concrete
ChatService/SearchService instantiation happens here (see api/app.py for the
composition root that wires concrete providers in).
"""
import json
from typing import Any, Dict, Generator, List

from core.config import RetrievalSettings
from core.interfaces import LLMProvider
from core.models import EventStatus, EventStep
from core.prompts import (
    SUB_QUERY_INSTRUCTIONS,
    SUB_QUERY_SCHEMA,
    SYSTEM_PROMPT,
    build_context_message,
    format_context,
    with_no_think,
)
from services.search import SemanticSearchService
from tools.registry import ToolRegistry


class RAGPipeline:
    def __init__(
        self,
        llm: LLMProvider,
        semantic_search: SemanticSearchService,
        tool_registry: ToolRegistry,
        retrieval_settings: RetrievalSettings,
    ):
        self.llm = llm
        self.semantic_search = semantic_search
        self.tool_registry = tool_registry
        self.settings = retrieval_settings

    def _deduplicate_docs(self, docs: List[Dict]) -> List[Dict]:
        """Remove duplicate documents based on chunk_id."""
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_id = doc.get("chunk_id") or hash(doc.get("content", ""))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    def _get_last_user_question(self, messages: List[Dict[str, str]]) -> str:
        """Get the user's most recent question, used for logging / fallback."""
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def process(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        allow_reasoning: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """Main processing pipeline.

        messages: conversation history as [{"role": "user"/"assistant", "content": "..."}]
        in chronological order; no need to include a system prompt (the pipeline adds its own).

        allow_reasoning: mặc định False (tắt "thinking mode" của model, hành vi
        cũ) — khi True, bỏ qua with_no_think() và bật enable_thinking cho các
        model self-host hỗ trợ (vd Qwen3), phục vụ toggle "cho phép suy luận
        sâu" trên UI.
        """
        conversation = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        if not conversation:
            yield {"step": EventStep.ANSWER, "status": EventStatus.ERROR, "data": {"error": "Không có nội dung hội thoại hợp lệ."}}
            return

        question = self._get_last_user_question(conversation)

        # ==========================================
        # STEP 1: SUB-QUERY (analyze the question, based on the WHOLE conversation)
        # ==========================================
        yield {"step": EventStep.SUB_QUERIES, "status": EventStatus.PROCESSING, "data": None}

        try:
            sub_query_response = ""
            sub_query_messages = [
                {"role": "system", "content": SYSTEM_PROMPT + SUB_QUERY_INSTRUCTIONS},
                *messages,
            ]
            if not allow_reasoning:
                sub_query_messages = with_no_think(sub_query_messages)
            for message in self.llm.chat(
                messages=sub_query_messages,
                response_format=SUB_QUERY_SCHEMA,
                stream=False,
                enable_thinking=allow_reasoning,
            ):
                sub_query_response = json.loads(message.content)

            sub_queries = sub_query_response.get("queries", [])
        except Exception as e:
            print(f"Lỗi khi parse sub-queries: {e}")
            sub_queries = [question]

        yield {"step": EventStep.SUB_QUERIES, "status": EventStatus.DONE, "data": {"queries": sub_queries}}

        # ==========================================
        # STEP 2: SEARCH (initial semantic search)
        # ==========================================
        yield {"step": EventStep.RETRIEVAL, "status": EventStatus.PROCESSING, "data": None}

        retrieved_docs = []
        for sq in sub_queries:
            try:
                docs = self.semantic_search.semantic_search(query=sq, top_k=self.settings.semantic_top_k)
                retrieved_docs.extend(docs)
            except Exception as e:
                print(f"Lỗi search cho query '{sq}': {e}")

        unique_docs = self._deduplicate_docs(retrieved_docs)
        context_docs = unique_docs[: self.settings.max_context_chunks]

        yield {"step": EventStep.RETRIEVAL, "status": EventStatus.DONE, "data": {"count": len(context_docs)}}
        citation_map: Dict[str, Any] = {str(i + 1): d for i, d in enumerate(context_docs)}

        # ==========================================
        # STEP 2.5: CONTEXT READY
        # ------------------------------------------
        # Emit citations/sources RIGHT AFTER retrieval finishes, BEFORE the
        # LLM starts answering — so the reference-document sidebar appears
        # early for the user to see while waiting for the LLM to generate an answer.
        #
        # IMPORTANT: do NOT use step="answer", status="done" here, since that
        # is the real "answer is complete" signal at the end of the stream —
        # reusing it would make the frontend think the answer is done right
        # from the start (while "text" doesn't exist yet) and could turn off
        # the streaming UI entirely. Use a separate "context_ready" step so
        # the frontend can update the sidebar without touching "answer" logic.
        # ==========================================
        yield {
            "step": EventStep.CONTEXT_READY,
            "status": EventStatus.DONE,
            "data": {"citations": citation_map, "sources": context_docs},
        }

        # ==========================================================
        # STEP 3+4 (COMBINED): LLM STREAM — decides on a tool call and
        # answers directly in the SAME call, per the reference approach.
        # Loops up to max_tool_iterations times if the LLM keeps calling tools.
        # ==========================================================
        # Structure: [system context + citation rules, ...the whole original conversation]
        # Kept multi-turn so the LLM understands the conversation correctly, instead of flattening it into 1 user message.
        llm_messages = [
            build_context_message(context_docs),
            *conversation,
        ]

        full_answer = ""
        tool_schemas = self.tool_registry.schemas()

        for iteration in range(self.settings.max_tool_iterations):
            did_tool_call = False

            # Buffer to accumulate tool_call argument fragments split across multiple chunks
            # key = the tool call's index in the response (OpenAI can return multiple tool_calls in parallel)
            tool_call_buffers: Dict[int, Dict[str, Any]] = {}

            try:
                response_stream = self.llm.chat(
                    messages=llm_messages if allow_reasoning else with_no_think(llm_messages),
                    tools=tool_schemas,
                    stream=True,
                    enable_thinking=allow_reasoning,
                )
            except Exception as e:
                yield {"step": EventStep.ANSWER, "status": EventStatus.ERROR, "data": {"error": str(e)}}
                return

            if iteration == 0:
                yield {"step": EventStep.TOOL_CALL, "status": EventStatus.PROCESSING, "data": None}
                yield {"step": EventStep.ANSWER, "status": EventStatus.START, "data": None}

            try:
                for chunk in response_stream:
                    # LLMProvider yields {"error": ...} instead of raising when an error happens mid-stream
                    if isinstance(chunk, dict) and "error" in chunk:
                        raise Exception(chunk["error"])
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta

                    # --- Direct answer (no tool needed) ---
                    if getattr(delta, "content", None):
                        piece = delta.content
                        full_answer += piece
                        yield {
                            "step": EventStep.ANSWER,
                            "status": EventStatus.STREAMING,
                            "data": {"chunk": piece, "citations": citation_map},
                        }

                    # --- Tool call (may arrive in small fragments) ---
                    if getattr(delta, "tool_calls", None):
                        did_tool_call = True
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_buffers:
                                tool_call_buffers[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                            buf = tool_call_buffers[idx]
                            if tc_delta.id:
                                buf["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                buf["name"] += tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                buf["arguments"] += tc_delta.function.arguments

            except Exception as e:
                yield {"step": EventStep.ANSWER, "status": EventStatus.ERROR, "data": {"error": str(e)}}
                return

            # If the LLM didn't call a tool this round -> answer is done, exit the loop
            if not did_tool_call:
                break

            # ---- Process the accumulated tool calls ----
            assistant_tool_calls = []
            for idx in sorted(tool_call_buffers.keys()):
                buf = tool_call_buffers[idx]
                assistant_tool_calls.append({
                    "id": buf["id"],
                    "type": "function",
                    "function": {"name": buf["name"], "arguments": buf["arguments"]},
                })

            # Add an assistant message with tool_calls to history (required by the OpenAI spec)
            llm_messages.append({"role": "assistant", "content": None, "tool_calls": assistant_tool_calls})

            for tc in assistant_tool_calls:
                tool = self.tool_registry.get(tc["function"]["name"])
                if tool is None:
                    # unknown tool, skip safely
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Tool không được hỗ trợ.",
                    })
                    continue

                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield {"step": EventStep.TOOL_CALL, "status": EventStatus.DETECTED, "data": {"args": args}}

                try:
                    extra_docs = tool.execute(args, question=question)
                except Exception as e:
                    print(f"Lỗi thực thi tool: {e}")
                    extra_docs = []
                    yield {"step": EventStep.TOOL_CALL, "status": EventStatus.ERROR, "data": {"error": str(e)}}

                if extra_docs:
                    context_docs = self._deduplicate_docs(context_docs + extra_docs)[: self.settings.max_context_chunks]
                    citation_map = {str(i + 1): d for i, d in enumerate(context_docs)}
                    yield {"step": EventStep.TOOL_CALL, "status": EventStatus.EXECUTED, "data": {"found_count": len(extra_docs)}}

                    # Context was just extended -> re-emit "context_ready" so
                    # the frontend updates the sidebar with the latest document list.
                    yield {
                        "step": EventStep.CONTEXT_READY,
                        "status": EventStatus.DONE,
                        "data": {"citations": citation_map, "sources": context_docs},
                    }

                    tool_result_content = (
                        f"Đã tìm thấy {len(extra_docs)} đoạn trích từ văn bản {args.get('doc_ref')}. "
                        f"Ngữ cảnh đầy đủ đã được cập nhật ở lượt tiếp theo."
                    )
                else:
                    yield {"step": EventStep.TOOL_CALL, "status": EventStatus.EXECUTED, "data": {"found_count": 0, "message": "Không tìm thấy thông tin"}}
                    tool_result_content = f"Không tìm thấy thông tin bổ sung trong văn bản {args.get('doc_ref')}."

                llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result_content})

            # Update the "context" for the next call by adding 1 new user
            # message containing the extended context, so the model actually
            # "sees" the newly fetched content (not just the plain notice
            # message above).
            llm_messages.append({
                "role": "user",
                "content": (
                    f"Đây là ngữ cảnh đầy đủ đã được cập nhật sau khi tra cứu thêm:\n\n"
                    f"{format_context(context_docs)}\n\n"
                    f"Hãy trả lời câu hỏi gốc: {question}\n"
                    f"Nhớ tuân thủ quy tắc trích dẫn [N] như đã nêu. Nếu vẫn còn thiếu thông tin quan trọng "
                    f"và cần tra cứu thêm văn bản khác, hãy tiếp tục gọi tool."
                ),
            })

            yield {"step": EventStep.TOOL_CALL, "status": EventStatus.DONE, "data": None}
            # loop again -> call the LLM again with the new context

        # ==========================================
        # DONE: emit the real answer/done signal
        # ==========================================
        yield {
            "step": EventStep.ANSWER,
            "status": EventStatus.DONE,
            "data": {
                "text": full_answer,
                "citations": citation_map,
                "sources": context_docs,
            },
        }
