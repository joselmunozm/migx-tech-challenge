from app.config import Settings
from app.graph import _build_chat_model


class FakeOpenRouterClient:
    def __init__(self) -> None:
        self.chat = self
        self.request = None

    def send(self, **request):
        self.request = request
        return {
            "model": "fallback-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        }


def test_chat_model_sends_openrouter_fallbacks_and_bounded_timeout() -> None:
    client = FakeOpenRouterClient()
    model = _build_chat_model(
        Settings(
            openrouter_api_key="test-key",
            openrouter_model="primary-model",
            openrouter_fallback_model="fallback-model",
        )
    )
    model.client = client

    response = model.invoke("Hi")

    assert response.content == "OK"
    assert client.request["model"] == "primary-model"
    assert client.request["models"] == ["primary-model", "fallback-model"]
    assert model.request_timeout == 30_000
    assert model.max_retries == 0


def test_chat_model_omits_blank_fallback_and_duplicate_models() -> None:
    model = _build_chat_model(
        Settings(
            openrouter_api_key="test-key",
            openrouter_model="primary-model",
            openrouter_fallback_model="primary-model",
        )
    )

    assert model._default_params["models"] == ["primary-model"]
