"""The single tool the legal RAG assistant currently exposes: looking up a
specific clause/article inside a cited legal document."""
from typing import Any, Dict, List

from services.search import DocRefSearchService

_SCHEMA = {
    "name": "search_referenced_document",
    "description": (
        "Tìm kiếm nội dung cụ thể trong một văn bản pháp luật được trích dẫn. "
        "Sử dụng KHI VÀ CHỈ KHI ngữ cảnh hiện tại nhắc đến một văn bản khác (vd: Luật X, Thông tư Y) "
        "và bạn BẮT BUỘC cần chi tiết từ văn bản đó để trả lời chính xác câu hỏi."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_ref": {
                "type": "string",
                "description": "Số hiệu văn bản pháp luật ĐẦY ĐỦ (ví dụ: '36/2015/QĐ-TTg'). KHÔNG điền [1], [2].",
            },
            "dieu_filter": {
                "type": "string",
                "description": "(Tùy chọn) Chỉ ghi số điều, ví dụ 'Điều 74'.",
            },
            "khoan_filter": {
                "type": "string",
                "description": "(Tùy chọn) Chỉ ghi số khoản, ví dụ 'Khoản 3'.",
            },
            "content_query": {
                "type": "string",
                "description": "(Bắt buộc) Từ khóa hoặc chủ đề cần tìm trong văn bản đó.",
            },
        },
        "required": ["doc_ref", "content_query"],
    },
}


class SearchReferencedDocumentTool:
    name = "search_referenced_document"
    schema = _SCHEMA

    def __init__(self, doc_ref_search_service: DocRefSearchService, top_k: int = 5):
        self._service = doc_ref_search_service
        self._top_k = top_k

    def execute(self, args: Dict[str, Any], *, question: str) -> List[dict]:
        return self._service.doc_ref_search(
            query=args.get("content_query", question),
            doc_ref=args.get("doc_ref"),
            article_filter=args.get("dieu_filter"),
            clause_filter=args.get("khoan_filter"),
            top_k=self._top_k,
        )
