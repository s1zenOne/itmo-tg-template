"""Сборка ограниченного контекста без разрыва пар реплик."""

from app.store import HistoryMessage


class ContextLimitError(ValueError):
    """Системная инструкция и новый запрос не помещаются в бюджет."""


def _estimated_chars(role: str, content: str) -> int:
    # Небольшой запас учитывает поля роли и оболочку сообщения; это не точные токены.
    return len(content) + len(role) + 16


def build_messages(
    system: str,
    history: list[HistoryMessage],
    current: str,
    *,
    max_history_messages: int,
    max_context_chars: int,
) -> list[dict[str, str]]:
    base_size = _estimated_chars("system", system) + _estimated_chars("user", current)
    if base_size > max_context_chars:
        raise ContextLimitError("Системная инструкция и запрос превышают бюджет контекста.")

    selected: list[tuple[HistoryMessage, HistoryMessage]] = []
    used = base_size
    index = len(history) - 1
    while index >= 1 and len(selected) * 2 < max_history_messages:
        user, assistant = history[index - 1], history[index]
        if user.role != "user" or assistant.role != "assistant":
            index -= 1
            continue
        turn_size = _estimated_chars(user.role, user.content) + _estimated_chars(
            assistant.role, assistant.content
        )
        if used + turn_size > max_context_chars:
            break
        selected.append((user, assistant))
        used += turn_size
        index -= 2

    messages = [{"role": "system", "content": system}]
    for user, assistant in reversed(selected):
        messages.extend(
            [
                {"role": user.role, "content": user.content},
                {"role": assistant.role, "content": assistant.content},
            ]
        )
    messages.append({"role": "user", "content": current})
    return messages
