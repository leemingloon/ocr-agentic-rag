"""
PaddleX (pulled by recent paddleocr wheels) imports legacy LangChain 0.1 modules:
  langchain.docstore.document, langchain.text_splitter

Those paths were removed in LangChain 1.x. Installing real legacy langchain conflicts
with this repo's langgraph stack. Minimal stubs unblock `from paddleocr import PaddleOCR`
for document OCR only (retriever code in paddlex is not used).

Call `install_paddle_langchain_shim()` before importing paddleocr anywhere.
"""


from __future__ import annotations

import sys
import types


def install_paddle_langchain_shim() -> None:
    if sys.modules.get("langchain.docstore.document") and sys.modules.get("langchain.text_splitter"):
        return

    doc_mod = types.ModuleType("langchain.docstore.document")

    class Document:
        __slots__ = ("page_content", "metadata")

        def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata if metadata is not None else {}

    doc_mod.Document = Document
    sys.modules.setdefault("langchain.docstore.document", doc_mod)

    ds_pkg = types.ModuleType("langchain.docstore")
    sys.modules.setdefault("langchain.docstore", ds_pkg)

    ts_mod = types.ModuleType("langchain.text_splitter")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def split_text(self, text: str) -> list[str]:
            return [text] if text else []

    ts_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules.setdefault("langchain.text_splitter", ts_mod)
