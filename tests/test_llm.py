from video_rag.generation.llm import OpenRouterLLM, OpenRouterRateLimitError


class FakeRateLimitError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-min.', 'metadata': {'headers': {'X-RateLimit-Reset': '9999999999999'}}}}"
        )
        self.status_code = 429
        self.body = {
            "error": {
                "message": "Rate limit exceeded: free-models-per-min.",
                "metadata": {
                    "headers": {
                        "X-RateLimit-Reset": "9999999999999",
                    }
                },
            }
        }


class FakeCompletions:
    def create(self, **kwargs):
        raise FakeRateLimitError()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_openrouter_llm_rewrites_rate_limit_error_to_user_friendly_message() -> None:
    llm = OpenRouterLLM(api_key="test-key")
    llm._client = FakeClient()

    try:
        llm.generate(messages=[{"role": "user", "content": "hello"}])
        assert False, "Expected OpenRouterRateLimitError"
    except OpenRouterRateLimitError as exc:
        message = str(exc)
        assert "Превышен лимит запросов OpenRouter для бесплатной модели." in message
        assert "Повторите попытку позже" in message
