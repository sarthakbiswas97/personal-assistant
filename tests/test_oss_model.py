"""Tests for OSS model backend."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

from src.models.base import Message

# Pre-mock transformers so oss_model can be imported without the real library
_mock_transformers = ModuleType("transformers")
_mock_transformers.AutoModelForCausalLM = MagicMock()
_mock_transformers.AutoTokenizer = MagicMock()
_mock_transformers.TextIteratorStreamer = MagicMock()
sys.modules.setdefault("transformers", _mock_transformers)

from src.models.oss_model import OSSModel  # noqa: E402


@pytest.fixture
def mock_model() -> MagicMock:
    mock = MagicMock()
    mock.device = "cpu"
    return mock


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    mock = MagicMock()
    mock.apply_chat_template.return_value = "<formatted>"
    mock_encoded = MagicMock()
    mock_encoded.input_ids.to.return_value = torch.tensor([[1, 2, 3]])
    mock.return_value = mock_encoded
    return mock


@pytest.fixture
def oss_model(
    monkeypatch: pytest.MonkeyPatch,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
) -> OSSModel:
    """Create an OSSModel with mocked internals."""
    import src.models.oss_model as mod

    monkeypatch.setattr(mod, "AutoModelForCausalLM", MagicMock(from_pretrained=MagicMock(return_value=mock_model)))
    monkeypatch.setattr(mod, "AutoTokenizer", MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer)))

    model = OSSModel(model_name="test/model")
    return model


class TestOSSModel:
    """Tests for OSSModel Qwen backend."""

    def test_init_sets_model_to_eval(self, oss_model: OSSModel, mock_model: MagicMock) -> None:
        mock_model.eval.assert_called_once()

    def test_prepare_input_applies_chat_template(
        self, oss_model: OSSModel, mock_tokenizer: MagicMock
    ) -> None:
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        oss_model._prepare_input(messages)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    async def test_generate_returns_string(
        self, oss_model: OSSModel, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        # Make generate feed tokens into the streamer then end
        def fake_generate(**kwargs):
            streamer = kwargs["streamer"]
            for text in ["Hello", " world"]:
                streamer.text_queue.put(text)
            streamer.text_queue.put(streamer.stop_signal)

        mock_model.generate.side_effect = fake_generate

        # Use the real TextIteratorStreamer behavior via mock
        import queue

        class FakeStreamer:
            def __init__(self, tokenizer, **kwargs):
                self.text_queue = queue.Queue()
                self.stop_signal = None

            def __iter__(self):
                while True:
                    val = self.text_queue.get()
                    if val is self.stop_signal:
                        break
                    yield val

            def put(self, value):
                self.text_queue.put(value)

            def end(self):
                self.text_queue.put(self.stop_signal)

        import src.models.oss_model as mod
        original_streamer = mod.TextIteratorStreamer
        mod.TextIteratorStreamer = FakeStreamer

        try:
            result = await oss_model.generate([Message(role="user", content="Hi")])
            assert isinstance(result, str)
            assert result == "Hello world"
        finally:
            mod.TextIteratorStreamer = original_streamer

    async def test_cleanup_deletes_model_and_tokenizer(
        self, oss_model: OSSModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.models.oss_model as mod

        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

        await oss_model.cleanup()

        assert not hasattr(oss_model, "_model")
        assert not hasattr(oss_model, "_tokenizer")
