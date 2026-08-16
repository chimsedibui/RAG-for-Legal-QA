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
        """Loại bỏ các document trùng lặp dựa trên chunk_id."""
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_id = doc.get("chunk_id") or hash(doc.get("content", ""))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    def _get_last_user_question(self, messages: List[Dict[str, str]]) -> str:
        """Lấy câu hỏi mới nhất của user, dùng để log / fallback."""
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def process(self, messages: List[Dict[str, str]], stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Pipeline xử lý chính.

        messages: lịch sử hội thoại dạng [{"role": "user"/"assistant", "content": "..."}]
        theo đúng thứ tự thời gian, không cần chứa system prompt (pipeline tự thêm).
        """
        conversation = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        if not conversation:
            yield {"step": EventStep.ANSWER, "status": EventStatus.ERROR, "data": {"error": "Không có nội dung hội thoại hợp lệ."}}
            return

        question = self._get_last_user_question(conversation)

        # ==========================================
        # BƯỚC 1: SUB-QUERY (Phân tích câu hỏi, dựa trên TOÀN BỘ hội thoại)
        # ==========================================
        yield {"step": EventStep.SUB_QUERIES, "status": EventStatus.PROCESSING, "data": None}

        try:
            sub_query_response = ""
            sub_query_messages = with_no_think([
                {"role": "system", "content": SYSTEM_PROMPT + SUB_QUERY_INSTRUCTIONS},
                *messages,
            ])
            for message in self.llm.chat(
                messages=sub_query_messages,
                response_format=SUB_QUERY_SCHEMA,
                stream=False,
            ):
                sub_query_response = json.loads(message.content)

            sub_queries = sub_query_response.get("queries", [])
        except Exception as e:
            print(f"Lỗi khi parse sub-queries: {e}")
            sub_queries = [question]

        yield {"step": EventStep.SUB_QUERIES, "status": EventStatus.DONE, "data": {"queries": sub_queries}}

        # ==========================================
        # BƯỚC 2: SEARCH (Semantic Search ban đầu)
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
        # BƯỚC 2.5: CONTEXT READY
        # ------------------------------------------
        # Trả ra citations/sources NGAY khi vừa retrieval xong, TRƯỚC khi
        # LLM bắt đầu trả lời — để sidebar tài liệu tham khảo hiện lên sớm
        # cho người dùng xem trong lúc chờ LLM sinh câu trả lời.
        #
        # QUAN TRỌNG: KHÔNG dùng step="answer", status="done" ở đây, vì đó
        # là tín hiệu "câu trả lời đã hoàn tất" thật sự ở cuối luồng — nếu
        # dùng trùng, frontend sẽ tưởng câu trả lời xong ngay từ đầu (trong
        # khi "text" chưa tồn tại) và có thể tắt luôn UI streaming.
        # Dùng step riêng "context_ready" để frontend cập nhật sidebar mà
        # không đụng vào logic xử lý "answer".
        # ==========================================
        yield {
            "step": EventStep.CONTEXT_READY,
            "status": EventStatus.DONE,
            "data": {"citations": citation_map, "sources": context_docs},
        }

        # ==========================================================
        # BƯỚC 3+4 (GỘP): LLM STREAM — vừa quyết định tool call vừa
        # trả lời trực tiếp trong CÙNG một lần gọi, giống code mẫu.
        # Lặp tối đa max_tool_iterations lần nếu LLM liên tục gọi tool.
        # ==========================================================
        # Cấu trúc: [system context+quy tắc trích dẫn, ...toàn bộ hội thoại gốc]
        # Giữ nguyên multi-turn để LLM hiểu đúng mạch hội thoại, thay vì gộp hết vào 1 user message.
        llm_messages = [
            build_context_message(context_docs),
            *conversation,
        ]

        full_answer = ""
        tool_schemas = self.tool_registry.schemas()

        for iteration in range(self.settings.max_tool_iterations):
            did_tool_call = False

            # Buffer để gom các mảnh tool_call arguments bị chia nhỏ qua nhiều chunk
            # key = index của tool call trong response (OpenAI có thể trả nhiều tool_calls song song)
            tool_call_buffers: Dict[int, Dict[str, Any]] = {}

            try:
                response_stream = self.llm.chat(
                    messages=with_no_think(llm_messages),
                    tools=tool_schemas,
                    stream=True,
                )
            except Exception as e:
                yield {"step": EventStep.ANSWER, "status": EventStatus.ERROR, "data": {"error": str(e)}}
                return

            if iteration == 0:
                yield {"step": EventStep.TOOL_CALL, "status": EventStatus.PROCESSING, "data": None}
                yield {"step": EventStep.ANSWER, "status": EventStatus.START, "data": None}

            try:
                for chunk in response_stream:
                    # LLMProvider yield {"error": ...} thay vì raise khi có lỗi ở giữa stream
                    if isinstance(chunk, dict) and "error" in chunk:
                        raise Exception(chunk["error"])
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta

                    # --- Trả lời trực tiếp (không cần tool) ---
                    if getattr(delta, "content", None):
                        piece = delta.content
                        full_answer += piece
                        yield {
                            "step": EventStep.ANSWER,
                            "status": EventStatus.STREAMING,
                            "data": {"chunk": piece, "citations": citation_map},
                        }

                    # --- Tool call (có thể tới theo từng mảnh nhỏ) ---
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

            # Nếu vòng này LLM không gọi tool -> đã trả lời xong, thoát loop
            if not did_tool_call:
                break

            # ---- Xử lý các tool call đã gom được ----
            assistant_tool_calls = []
            for idx in sorted(tool_call_buffers.keys()):
                buf = tool_call_buffers[idx]
                assistant_tool_calls.append({
                    "id": buf["id"],
                    "type": "function",
                    "function": {"name": buf["name"], "arguments": buf["arguments"]},
                })

            # Thêm assistant message chứa tool_calls vào history (bắt buộc theo chuẩn OpenAI)
            llm_messages.append({"role": "assistant", "content": None, "tool_calls": assistant_tool_calls})

            for tc in assistant_tool_calls:
                tool = self.tool_registry.get(tc["function"]["name"])
                if tool is None:
                    # tool lạ, bỏ qua an toàn
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

                    # Context vừa được bổ sung -> phát lại "context_ready" để
                    # frontend cập nhật sidebar với danh sách tài liệu mới nhất.
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

            # Cập nhật lại phần "ngữ cảnh" cho lượt gọi tiếp theo bằng cách
            # thêm 1 user message mới chứa context đã bổ sung, để model
            # thực sự "nhìn thấy" nội dung mới lấy được (không chỉ là message
            # thông báo suông ở trên).
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
            # loop tiếp -> gọi lại LLM với context mới

        # ==========================================
        # KẾT THÚC: phát tín hiệu answer/done thật sự
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
