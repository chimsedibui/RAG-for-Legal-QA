"""Prompt text and prompt-shaping helpers for the legal RAG pipeline.

Kept separate from services/rag_pipeline.py so prompt content can be reviewed
and changed independently of orchestration logic.
"""
from typing import Any, Dict, List

NO_THINK_SUFFIX = "/no_think"

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý thông minh. Hãy trả lời chính xác, chuyên nghiệp dựa trên ngữ cảnh được cung cấp. "
    "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin."
)

SUB_QUERY_INSTRUCTIONS = (
    "Hãy phân tích đoạn hội thoại sau, tập trung vào ý định mới nhất của người dùng, "
    "và tách câu hỏi thành các sub-queries để tìm kiếm thông tin hiệu quả hơn.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. Mỗi sub-query PHẢI là một câu hỏi ĐẦY ĐỦ NGỮ CẢNH, hiểu được độc lập mà không cần đọc "
    "các sub-query khác. Nếu câu hỏi gốc có chủ thể/điều kiện chung (trình độ chuyên môn, loại "
    "hợp đồng, mốc thời gian, đối tượng áp dụng...), BẮT BUỘC phải LẶP LẠI (chèn lại) điều kiện đó "
    "vào TỪNG sub-query được tách ra — không được lược bỏ dù đã nêu ở sub-query trước.\n"
    "2. Chỉ tách thành nhiều sub-query khi các ý có thể tìm kiếm ĐỘC LẬP mà không mất nghĩa "
    "(vd: hai chủ đề pháp lý khác nhau, không chia sẻ chung điều kiện/chủ thể).\n"
    "3. Nếu câu hỏi gốc chỉ có MỘT ý chính, hoặc các ý nhỏ gắn chặt với nhau và không thể tách rời "
    "mà vẫn giữ đủ nghĩa, hãy trả về DUY NHẤT 1 sub-query giống với câu hỏi gốc (diễn đạt lại rõ hơn "
    "nếu cần) thay vì cố tách ra nhiều ý.\n\n"
    "Trả về kết quả dưới dạng JSON thuần túy với key 'queries'.\n\n"
)

SUB_QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sub_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "suy luận ngắn gọn"},
                "queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["queries"],
            "additionalProperties": True,
        },
    },
}


def format_context(docs: List[Dict[str, Any]]) -> str:
    """Format context thành dạng [1]: content, [2]: content..."""
    if not docs:
        return "Không có thông tin ngữ cảnh nào."
    return "\n\n".join(f"[{i + 1}]: {d.get('content', '')}" for i, d in enumerate(docs))


def build_context_message(context_docs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Tạo 1 system message chứa ngữ cảnh + quy tắc trích dẫn, được chèn vào
    NGAY TRƯỚC lượt hội thoại của user để LLM luôn thấy context mới nhất mà
    không phá vỡ cấu trúc nhiều lượt hội thoại."""
    context_text = format_context(context_docs)
    return {
        "role": "system",
        "content": f"""Bạn là trợ lý pháp lý thông minh. Hãy trả lời chính xác, chuyên nghiệp dựa trên ngữ cảnh được cung cấp.
            Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin.
            Dựa trên ngữ cảnh pháp lý sau để trả lời câu hỏi mới nhất của người dùng trong hội thoại:
{context_text}

QUY TẮC TRÍCH DẪN BẮT BUỘC:
- Mọi thông tin lấy từ ngữ cảnh đều phải trích dẫn nguồn.
- Sử dụng CHÍNH XÁC định dạng [N] (ví dụ: [1], [2], [3]).
- KHÔNG thêm khoảng trắng (không dùng [ 1 ]), KHÔNG dùng định dạng khác.
- Đặt mã trích dẫn ở cuối câu hoặc cuối ý tương ứng.
- Nếu ngữ cảnh NHẮC ĐẾN một văn bản khác (vd: "theo Luật X") và bạn CẦN chi tiết từ văn bản đó để trả lời
  chính xác, hãy gọi tool `search_referenced_document` thay vì trả lời ngay.
- Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin.
- Hãy tham khảo các lượt hội thoại trước đó (nếu có) để hiểu đúng ý người dùng, nhưng chỉ trích dẫn [N]
  cho thông tin lấy từ ngữ cảnh pháp lý ở trên.""",
    }


def with_no_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trả về 1 list MỚI: với mỗi message role='user' có content dạng string
    mà chưa kết thúc bằng NO_THINK_SUFFIX, nối thêm suffix đó vào cuối (dùng
    để tắt "thinking mode" của các model Qwen3 — no-op vô hại với model khác).

    KHÔNG mutate list/dict đầu vào, để lịch sử hội thoại gốc (llm_messages)
    giữ nguyên qua nhiều vòng lặp tool-call mà không bị cộng dồn suffix.
    """
    result = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, str) and not content.rstrip().endswith(NO_THINK_SUFFIX):
            result.append({**m, "content": f"{content.rstrip()} {NO_THINK_SUFFIX}"})
        else:
            result.append(m)
    return result
