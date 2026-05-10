from video_rag.generation.prompts import (
    build_reduce_summary_messages,
    build_retrieval_query,
    build_answer_messages,
    build_summary_messages,
    infer_response_language,
    limit_history,
    normalize_answer_text,
    normalize_summary_structure,
)


def test_limit_history_keeps_last_turns() -> None:
    history = [
        ("q1", "a1"),
        ("q2", "a2"),
        ("q3", "a3"),
    ]
    assert limit_history(history, 2) == [("q2", "a2"), ("q3", "a3")]


def test_build_retrieval_query_returns_plain_question_without_history() -> None:
    assert build_retrieval_query("Когда обсуждали бюджет?") == "Когда обсуждали бюджет?"


def test_infer_response_language_uses_question_language() -> None:
    assert infer_response_language("О чем это видео?") == "русском"
    assert infer_response_language("What is this video about?") == "английском"


def test_normalize_summary_structure_fills_missing_sections() -> None:
    summary = normalize_summary_structure("Тезисы\n- Обсудили бюджет [00:10]")

    assert "## Тезисы" in summary
    assert "## Решения" in summary
    assert "## Таймлайн" in summary
    assert "- Не указано." in summary


def test_normalize_summary_structure_drops_meta_leak_lines() -> None:
    summary = normalize_summary_structure(
        "Тезисы\nWe need to produce final answer\n- Real fact [00:10]"
    )

    assert "We need to produce final answer" not in summary
    assert "- Real fact" in summary
    assert "[00:10]" not in summary


def test_build_answer_messages_binds_language_to_question_not_context() -> None:
    messages = build_answer_messages(
        question="О чем это видео?",
        context="The video is about AI assistants and automation.",
    )

    assert "Язык ответа определяется только языком вопроса пользователя" in messages[0]["content"]
    assert "полностью на русском языке" in messages[0]["content"]
    assert "переведи содержание ответа на русском язык" in messages[-1]["content"]


def test_normalize_answer_text_cleans_common_english_leakage_for_russian_question() -> None:
    normalized = normalize_answer_text(
        "O mention UncleFrank произносится в fragment#1 и repeats in the same section around [04:55].",
        question="когда в видео говорят о дяде френке",
    )

    assert "mention" not in normalized.lower()
    assert "fragment#1" not in normalized.lower()
    assert "repeats" not in normalized.lower()
    assert "around" not in normalized.lower()
    assert "упоминание UncleFrank" in normalized
    assert "фрагменте #1" in normalized


def test_build_summary_messages_uses_russian_for_whole_video_summary() -> None:
    messages = build_summary_messages("The transcript is fully in English.")

    assert "полностью на русском языке" in messages[0]["content"]
    assert "Даже если контекст полностью на другом языке" in messages[-1]["content"]


def test_build_summary_messages_requires_timeline_only_timecodes() -> None:
    messages = build_summary_messages("Context")

    assert "Таймкоды указывай только в разделе Таймлайн" in messages[-1]["content"]
    assert "Пункты действий" not in messages[-1]["content"]
    assert "Англоязычные термины адаптируй в понятные русские формулировки" in messages[-1]["content"]


def test_build_reduce_summary_messages_uses_russian_for_whole_video_summary() -> None:
    messages = build_reduce_summary_messages(["## Тезисы\n- A fact"])

    assert "только на русском языке" in messages[0]["content"]
    assert "на русском языке" in messages[-1]["content"]
    assert "Даже если промежуточные конспекты или исходный контекст на другом языке" in messages[0]["content"]


def test_normalize_summary_structure_parses_inline_sections() -> None:
    summary = normalize_summary_structure(
        "Тезисы: - Первый факт. Решения: - Второй факт. Пункты действий: - Третий факт. Таймлайн: - [00:10-00:20] Эпизод."
    )

    assert "## Тезисы\n- Первый факт." in summary
    assert "## Решения\n- Второй факт.\n- Третий факт." in summary
    assert "## Таймлайн\n- [00:10-00:20] Эпизод." in summary


def test_normalize_summary_structure_drops_reduce_meta_phrases() -> None:
    summary = normalize_summary_structure(
        "From first intermediate: Тезисы:\n- Факт 1\nNow combine\nРешения: - Факт 2"
    )

    assert "From first intermediate" not in summary
    assert "Now combine" not in summary
    assert "## Тезисы" in summary
    assert "## Решения" in summary


def test_normalize_summary_structure_treats_bare_section_lines_as_headings() -> None:
    summary = normalize_summary_structure(
        "Тезисы\nТезисы\n- Факт 1\nРешения\n- Факт 2\nПункты действий\n- Факт 3"
    )

    assert summary.count("## Тезисы") == 1
    assert "## Тезисы\n- Факт 1" in summary
    assert "## Решения\n- Факт 2\n- Факт 3" in summary


def test_normalize_summary_structure_removes_timecodes_outside_timeline() -> None:
    summary = normalize_summary_structure(
        "Тезисы\n- Факт [02:24-03:17]\nТаймлайн\n- [02:24-03:17] Эпизод"
    )

    assert "## Тезисы\n- Факт" in summary
    assert "## Таймлайн\n- [02:24-03:17] Эпизод" in summary


def test_normalize_summary_structure_drops_truncation_noise_and_cjk() -> None:
    summary = normalize_summary_structure(
        "Тезисы\n- Контекст обрывается ([03:39-04:27])\n- Пример: 勃起disfunción\nТаймлайн\n- [03:39-04:27] Финал"
    )

    assert "контекст обрывается" not in summary.lower()
    assert "勃起" not in summary
    assert "## Таймлайн\n- [03:39-04:27] Финал" in summary


def test_normalize_summary_structure_treats_none_as_empty_section() -> None:
    summary = normalize_summary_structure("Тезисы\nNone")

    assert "## Тезисы\n- Не указано." in summary
    assert "\nNone\n" not in summary
