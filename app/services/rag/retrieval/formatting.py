from __future__ import annotations

from app.services.rag.retrieval.retriever import ChunkResult, SearchResult


def format_search_result(result: SearchResult) -> str:
    """
    Сериализует SearchResult в текст для модели.
    """
    if not result.top_chunks and not result.group_chunks and not result.mentioned_chunks:
        return "Ничего не найдено."

    # Собираем порядок групп по parent_id
    all_chunks = result.top_chunks + result.group_chunks

    group_order: list[str] = []
    groups: dict[str, list[ChunkResult]] = {}
    for chunk in all_chunks:
        parent_id = chunk.metadata.get("parent_id", chunk.id)
        if parent_id not in groups:
            group_order.append(parent_id)
            groups[parent_id] = []
        groups[parent_id].append(chunk)

    # Сортируем по inner_id внутри группы
    def inner_id_key(c: ChunkResult) -> int:
        try:
            return int(c.metadata.get("inner_id", 0))
        except (ValueError, TypeError):
            return 0

    ordered_chunks: list[ChunkResult] = []
    for parent_id in group_order:
        ordered_chunks.extend(sorted(groups[parent_id], key=inner_id_key))

    # Группируем по section_header, сохраняя порядок
    parts: list[str] = []
    current_section: str | None = None
    section_lines: list[str] = []

    def flush_section() -> None:
        if current_section is not None and section_lines:
            parts.append(f"{current_section}")
            parts.extend(section_lines)

    for chunk in ordered_chunks:
        section = chunk.section_header or "Без раздела"
        chunk_text = chunk.text.strip()

        if section != current_section:
            flush_section()
            current_section = section
            section_lines = [chunk_text]
        else:
            section_lines.append(chunk_text)

    flush_section()

    # Добавляем упомянутые чанки
    if result.mentioned_chunks:
        parts.append("Упомянутые чанки:")
        for chunk in result.mentioned_chunks:
            label = result.mentioned_labels.get(chunk.id, "")
            header = f"{label}:" if label else "—"
            parts.append(f"{header}\n{chunk.text.strip()}")

    return "\n\n".join(parts)