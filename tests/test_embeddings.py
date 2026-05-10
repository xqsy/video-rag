from video_rag.ingest.embeddings import EmbeddingEncoder


def test_e5_prefixes_are_applied() -> None:
    assert EmbeddingEncoder.format_query("когда был бюджет?") == "query: когда был бюджет?"
    assert EmbeddingEncoder.format_passage("Бюджет обсуждали в начале") == "passage: Бюджет обсуждали в начале"
