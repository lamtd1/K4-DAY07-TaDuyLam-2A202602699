from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Empty store: nothing to ground an answer on, so do not call the LLM.
            return "Không tìm thấy thông tin nào trong cơ sở tri thức để trả lời câu hỏi này."

        context = "\n\n".join(
            f"[{i}] (nguồn: {self._source_of(result)})\n{result['content']}"
            for i, result in enumerate(results, start=1)
        )
        prompt = (
            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu.\n"
            "Chỉ dùng thông tin trong phần NGỮ CẢNH bên dưới. Không dùng kiến thức bên ngoài và không suy đoán.\n"
            "Nếu ngữ cảnh không có thông tin để trả lời, hãy nói rõ là không tìm thấy thông tin.\n"
            "Khi trả lời, trích dẫn số đoạn nguồn bạn dùng, ví dụ [1] hoặc [2].\n"
            "Trả lời bằng cùng ngôn ngữ với câu hỏi.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )
        return self.llm_fn(prompt)

    @staticmethod
    def _source_of(result: dict) -> str:
        """Best available pointer back to the original document for a retrieved chunk."""
        metadata = result.get("metadata") or {}
        for key in ("source_url", "source", "doc_id"):
            if metadata.get(key):
                return f"{metadata[key]} (chunk {result['id']})"
        return f"chunk {result['id']}"
