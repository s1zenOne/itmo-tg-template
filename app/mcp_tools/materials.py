"""Небольшой локальный корпус: фиксированный реестр и ограниченный поиск."""

import asyncio
import re
from pathlib import Path

from pydantic import ValidationError

from app.mcp_tools.contracts import MaterialEntry, MaterialsCatalog, MaterialsResult, ToolError

DEFAULT_PATH = Path(__file__).with_name("data") / "materials"
# IDs are public references, not paths supplied by a model or a user.
DOCUMENTS = (
    ("mcp-introduction", "MCP и инструменты", "mcp.md"),
    ("context-and-output", "Контекстное окно и структурированный вывод", "context.md"),
    ("idempotent-delivery", "Идемпотентность и доставка", "idempotency.md"),
)
_STOPWORDS = frozenset(
    {
        "какие",
        "какой",
        "что",
        "это",
        "как",
        "для",
        "при",
        "или",
        "если",
        "свой",
        "учебным",
        "материалам",
        "материалах",
        "материал",
        "конспекте",
        "документам",
        "объясни",
        "объяснение",
        "расскажи",
        "найди",
        "составь",
        "краткое",
        "относится",
        "относятся",
        "правила",
        "каких",
    }
)
_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _terms(text: str) -> set[str]:
    words = _WORD.findall(text.casefold())
    return {word[:5] for word in words if len(word) >= 3 and word not in _STOPWORDS}


class FileMaterials:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path

    def _documents(self) -> list[MaterialEntry]:
        documents = []
        try:
            for identifier, title, filename in DOCUMENTS:
                with (self.path / filename).open("rb") as file:
                    data = file.read(16001)
                if len(data) > 16000:
                    raise ToolError("materials_unavailable")
                documents.append(
                    MaterialEntry(id=identifier, title=title, text=data.decode("utf-8"))
                )
        except (OSError, UnicodeError, ValidationError):
            raise ToolError("materials_unavailable") from None
        return documents

    async def catalog(self) -> MaterialsCatalog:
        documents = await asyncio.to_thread(self._documents)
        return MaterialsCatalog(
            documents=[{"id": item.id, "title": item.title} for item in documents]
        )

    async def search(self, query: str, limit: int = 5) -> MaterialsResult:
        documents = await asyncio.to_thread(self._documents)
        terms = _terms(query)
        ranked = [(len(terms & _terms(item.title + " " + item.text)), item) for item in documents]
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        return MaterialsResult(
            status="ok", documents=[item for score, item in ranked if score][:limit]
        )
