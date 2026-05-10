from __future__ import annotations

from collections.abc import Sequence
import re

from video_rag.schemas import Chunk, RetrievalResult


SUMMARY_SECTION_TITLES = ("Тезисы", "Решения", "Таймлайн")
META_LEAK_PATTERNS = (
    "we need to",
    "the user",
    "use only provided context",
    "actually the conversation",
    "the latest user message",
    "thus we need to",
    "possibly they",
    "let's extract facts",
    "instruction says",
    "from first intermediate",
    "second intermediate",
    "now combine",
    "let's merge",
    "that's combined",
)
SECTION_MARKER_PATTERN = re.compile(
    r"(?i)(тезисы|решения|пункты действий|действия|action items|action-items|таймлайн|timeline)\s*:"
)
TIMECODE_RANGE_PATTERN = re.compile(r"\[(?:\d{2}:)?\d{2}:\d{2}(?:-(?:\d{2}:)?\d{2}:\d{2})?\]")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SUMMARY_NOISE_PATTERNS = (
    re.compile(r"\(?контекст\s+обрывается\)?", re.IGNORECASE),
    re.compile(r"\(?контекст\s+прерывается\)?", re.IGNORECASE),
    re.compile(r"\(?context\s+(?:is\s+)?truncated\)?", re.IGNORECASE),
    re.compile(r"\(?context\s+cuts\s+off\)?", re.IGNORECASE),
    re.compile(r"\(?context\s+ends\s+abruptly\)?", re.IGNORECASE),
)
SUMMARY_EMPTY_VALUES = {"none", "null", "n/a", "na", "nil"}
RUSSIAN_ANSWER_REPLACEMENTS = (
    (r"\bmention\b", "упоминание"),
    (r"\bmentioned\b", "упомянуто"),
    (r"\bfragment\s*#\s*(\d+)", r"фрагменте #\1"),
    (r"\bfragment#\s*(\d+)", r"фрагменте #\1"),
    (r"\brepeats\b", "повторяется"),
    (r"\baround\b", "около"),
    (r"\bin the same section\b", "в том же участке"),
    (r"\bin the range\b", "в диапазоне"),
)


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_retrieval_context(results: Sequence[RetrievalResult]) -> str:
    lines: list[str] = []
    for result in results:
        chunk = result.chunk
        lines.append(
            "\n".join(
                [
                    f"Фрагмент #{result.rank}",
                    f"Таймкод: [{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}]",
                    f"Релевантность: {result.score:.4f}",
                    f"Текст: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(lines)


def render_chunk_context(chunks: Sequence[Chunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            "\n".join(
                [
                    f"Чанк #{index}",
                    f"Таймкод: [{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}]",
                    f"Текст: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(lines)


def limit_history(history: Sequence[tuple[str, str]] | None, max_turns: int) -> list[tuple[str, str]]:
    if not history or max_turns <= 0:
        return []
    return list(history[-max_turns:])


def build_retrieval_query(
    question: str,
    history: Sequence[tuple[str, str]] | None = None,
    max_turns: int = 3,
) -> str:
    limited_history = limit_history(history, max_turns)
    if not limited_history:
        return question

    lines = ["Текущий вопрос: " + question, "", "Последние реплики диалога:"]
    for user_message, assistant_message in limited_history:
        lines.append(f"- Пользователь: {user_message}")
        lines.append(f"- Ассистент: {assistant_message}")
    return "\n".join(lines)


def render_history(history: Sequence[tuple[str, str]] | None) -> str:
    if not history:
        return ""

    history_lines: list[str] = []
    for user_message, assistant_message in history:
        history_lines.append(f"Пользователь: {user_message}")
        history_lines.append(f"Ассистент: {assistant_message}")
    return "История диалога:\n" + "\n".join(history_lines) + "\n\n"


def infer_response_language(text: str | None) -> str:
    value = (text or "").strip()
    if not value:
        return "русском"
    if re.search(r"[А-Яа-яЁё]", value):
        return "русском"
    if re.search(r"[A-Za-z]", value):
        return "английском"
    return "русском"


def _looks_like_meta_leak(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return False
    return any(pattern in normalized for pattern in META_LEAK_PATTERNS)


def _normalize_section_name(value: str) -> str | None:
    normalized = value.strip().lower().rstrip(":")
    mapping = {
        "тезисы": "Тезисы",
        "решения": "Решения",
        "action items": "Решения",
        "action-items": "Решения",
        "действия": "Решения",
        "пункты действий": "Решения",
        "таймлайн": "Таймлайн",
        "timeline": "Таймлайн",
    }
    return mapping.get(normalized)


def _extract_section_heading(line: str) -> str | None:
    normalized = re.sub(r"^[#\-\d.\s]+", "", line).strip()
    return _normalize_section_name(normalized)


def _split_inline_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(SECTION_MARKER_PATTERN.finditer(text))
    if not matches:
        return [(None, text)]

    chunks: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            chunks.append((None, prefix))

    for index, match in enumerate(matches):
        section_name = _normalize_section_name(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        chunks.append((section_name, content))
    return chunks


def _clean_summary_line(line: str, section_title: str, response_language: str) -> str:
    cleaned = line.strip()
    had_bullet = bool(re.match(r"^[-*•]\s*", cleaned))
    cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
    for pattern in SUMMARY_NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if section_title != "Таймлайн":
        cleaned = TIMECODE_RANGE_PATTERN.sub("", cleaned)
    if response_language == "русском":
        cleaned = CJK_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:]){2,}", r"\1", cleaned)
    cleaned = cleaned.strip()
    if cleaned.lower() in SUMMARY_EMPTY_VALUES:
        return ""
    if not cleaned or re.fullmatch(r"[-—.,;:()\s]+", cleaned):
        return ""
    return f"- {cleaned}" if had_bullet else cleaned


def normalize_summary_structure(text: str, response_language: str = "русском") -> str:
    stripped = text.strip()
    sections: dict[str, list[str]] = {title: [] for title in SUMMARY_SECTION_TITLES}
    current_section = "Тезисы"

    for raw_line in stripped.splitlines() if stripped else []:
        line = raw_line.strip()
        if _looks_like_meta_leak(line):
            continue
        direct_section = _extract_section_heading(line)
        if direct_section is not None:
            current_section = direct_section
            continue
        for section_name, content in _split_inline_sections(line):
            if section_name is not None:
                current_section = section_name
            if not content:
                continue
            sections[current_section].append(content.rstrip())

    if not stripped:
        sections["Тезисы"].append("- Нет данных.")

    blocks: list[str] = []
    for title in SUMMARY_SECTION_TITLES:
        body_lines = [
            cleaned
            for line in sections[title]
            if (cleaned := _clean_summary_line(line, title, response_language))
        ]
        body = "\n".join(body_lines).strip() or "- Не указано."
        blocks.append(f"## {title}\n{body}")
    return "\n\n".join(blocks)


def normalize_answer_text(text: str, question: str) -> str:
    normalized = text.strip()
    if infer_response_language(question) != "русском":
        return normalized

    for pattern, replacement in RUSSIAN_ANSWER_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"\b([A-Za-z])\s+упоминание\b", "упоминание", normalized)
    return normalized.strip()


def build_answer_messages(
    question: str,
    context: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    history_block = render_history(history)
    response_language = infer_response_language(question)

    system_prompt = (
        "Ты ассистент по анализу видео. Отвечай только на основе предоставленного контекста. "
        "Если данных недостаточно, прямо скажи об этом. Таймкоды указывай только тогда, когда в контексте "
        "есть подходящий подтверждающий фрагмент. Используй формат [mm:ss] или [hh:mm:ss]. "
        f"Язык ответа определяется только языком вопроса пользователя. Независимо от языка аудио, транскрипта или контекста, "
        f"итоговый ответ должен быть полностью на {response_language} языке. "
        "Не выдумывай факты и не ссылайся на моменты, которых нет в контексте. "
        "Не используй английские слова и фразы в тексте ответа, кроме имён собственных, названий продуктов, брендов и API. "
        "Не пересказывай инструкции, не описывай ход рассуждений и не упоминай prompt, context или пользователя как часть ответа."
    )
    user_prompt = (
        f"{history_block}"
        f"Контекст:\n{context}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        f"Дай краткий и точный ответ на {response_language} языке. "
        f"Если контекст написан на другом языке, переведи содержание ответа на {response_language} язык, сохраняя имена собственные и таймкоды. "
        "Если в контексте нет ответа, прямо сообщи об этом и не добавляй таймкоды. "
        "Не повторяй служебные инструкции и не объясняй, как ты строил ответ."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(context: str) -> list[dict[str, str]]:
    response_language = "русском"
    system_prompt = (
        "Ты ассистент по анализу видео. Составляй структурированный конспект только по переданному контексту. "
        "Используй таймкоды в формате [mm:ss] или [hh:mm:ss] только в разделе Таймлайн. "
        "В разделах Тезисы и Решения не добавляй таймкоды. Не выдумывай детали. "
        f"Все разделы и пункты итогового конспекта должны быть полностью на {response_language} языке. "
        "Не пиши, что контекст обрывается, неполный или truncated; просто опускай неполные детали. "
        "Не используй иностранные фразы и иероглифы, кроме имён собственных, брендов, названий продуктов, API и файлов. "
        "Если в исходном материале встречаются термины на английском, подбирай для них понятные русские эквиваленты. "
        "Если без исходного термина можно сохранить смысл, не оставляй английское слово. Если английский термин важен, сначала дай русский эквивалент, а оригинал оставь только один раз в скобках. "
        "Всегда возвращай markdown с заголовками второго уровня: ## Тезисы, ## Решения, ## Таймлайн. "
        "Не выводи служебные инструкции, не пересказывай prompt и не показывай рассуждения о задаче."
    )
    user_prompt = (
        f"Контекст:\n{context}\n\n"
        f"Сформируй структурированный конспект на {response_language} языке в 3 разделах: "
        "Тезисы, Решения, Таймлайн. Для каждого раздела дай краткие пункты. "
        f"Даже если контекст полностью на другом языке, переведи содержательные пункты на {response_language} язык, сохраняя имена собственные, технические термины при необходимости и таймкоды. "
        "Англоязычные термины адаптируй в понятные русские формулировки. Если термин важен сам по себе, сначала укажи русский вариант, а английский оставь только как краткую подсказку в скобках. "
        "Таймкоды указывай только в разделе Таймлайн. В остальных разделах давай формулировки без таймкодов. "
        "Не пиши фразы про обрыв или неполноту контекста. Не используй иероглифы и иностранные вставки, если это не имя собственное или название. "
        "Сразу начни с заголовка раздела. Не пиши фразы вроде 'We need to', 'The user asked' или 'Use only provided context'. "
        "Если для раздела нет данных, напиши '- Не указано.'"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_reduce_summary_messages(partial_summaries: Sequence[str]) -> list[dict[str, str]]:
    response_language = "русском"
    system_prompt = (
        "Ты ассистент по анализу видео. Объедини промежуточные конспекты в единый итоговый конспект. "
        "Сохраняй только подтверждённые факты и удаляй повторы. "
        "Таймкоды оставляй только в разделе Таймлайн. В разделах Тезисы и Решения таймкоды не нужны. "
        "Всегда возвращай markdown с заголовками второго уровня: ## Тезисы, ## Решения, ## Таймлайн. "
        f"Даже если промежуточные конспекты или исходный контекст на другом языке, итоговый конспект должен быть полностью на {response_language} языке. "
        f"Пиши итоговый конспект только на {response_language} языке. "
        "Не пиши, что контекст обрывается, неполный или truncated; просто опускай неполные детали. "
        "Не используй иностранные фразы и иероглифы, кроме имён собственных, брендов, названий продуктов, API и файлов. "
        "Англоязычные термины переводи в понятные русские эквиваленты. Если оригинальный термин действительно нужен, сначала дай русский вариант, а английский оставь только один раз в скобках. "
        "Не выводи инструкции, не описывай процесс анализа и не пиши мета-комментарии о запросе или контексте."
    )
    user_prompt = (
        "Промежуточные конспекты:\n\n"
        + "\n\n---\n\n".join(partial_summaries)
        + f"\n\nСобери финальный конспект на {response_language} языке в разделах: Тезисы, Решения, Таймлайн."
        + f" Сразу начни с заголовка раздела. Для всех пояснений и пунктов используй только этот язык. Переведи содержательные пункты на {response_language} язык, даже если промежуточные конспекты написаны на другом языке."
        + " Адаптируй англоязычные термины в понятные русские формулировки; если исходный термин важен, поставь его только после русского эквивалента в скобках."
        + " Таймкоды оставляй только в разделе Таймлайн. Не пиши фразы про обрыв или неполноту контекста. Не используй иероглифы и иностранные вставки, если это не имя собственное или название."
        + " Не повторяй служебные формулировки и не объясняй, как ты объединял данные."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
