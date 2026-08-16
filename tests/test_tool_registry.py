from tools.doc_ref_tool import SearchReferencedDocumentTool
from tools.registry import ToolRegistry


class _FakeTool:
    name = "fake_tool"
    schema = {"name": "fake_tool", "description": "d", "parameters": {}}

    def execute(self, args, *, question):
        return [{"chunk_id": "x"}]


def test_register_and_get():
    registry = ToolRegistry()
    tool = _FakeTool()
    registry.register(tool)
    assert registry.get("fake_tool") is tool


def test_get_unknown_tool_returns_none():
    registry = ToolRegistry()
    assert registry.get("does_not_exist") is None


def test_schemas_shape():
    registry = ToolRegistry()
    registry.register(_FakeTool())
    schemas = registry.schemas()
    assert schemas == [{"type": "function", "function": _FakeTool.schema}]


class _FakeDocRefSearchService:
    def __init__(self):
        self.calls = []

    def doc_ref_search(self, query, doc_ref, article_filter=None, clause_filter=None, top_k=10):
        self.calls.append({
            "query": query, "doc_ref": doc_ref,
            "article_filter": article_filter, "clause_filter": clause_filter, "top_k": top_k,
        })
        return [{"chunk_id": "found"}]


def test_search_referenced_document_tool_maps_args_to_service_kwargs():
    service = _FakeDocRefSearchService()
    tool = SearchReferencedDocumentTool(service, top_k=7)

    args = {
        "doc_ref": "36/2015/QĐ-TTg",
        "dieu_filter": "Điều 74",
        "khoan_filter": "Khoản 3",
        "content_query": "thời hạn",
    }
    result = tool.execute(args, question="fallback question")

    assert result == [{"chunk_id": "found"}]
    assert service.calls == [{
        "query": "thời hạn",
        "doc_ref": "36/2015/QĐ-TTg",
        "article_filter": "Điều 74",
        "clause_filter": "Khoản 3",
        "top_k": 7,
    }]


def test_search_referenced_document_tool_falls_back_to_question():
    service = _FakeDocRefSearchService()
    tool = SearchReferencedDocumentTool(service)

    tool.execute({"doc_ref": "1/2020/ND-CP"}, question="fallback question")
    assert service.calls[0]["query"] == "fallback question"
