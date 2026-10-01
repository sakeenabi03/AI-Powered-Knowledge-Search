"""Unit tests for offline-capable EmbeddingService (mocked model loading)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import get_settings
from app.services.embedding_service import (
    EmbeddingService,
    _looks_like_sentence_transformer_model,
    get_embedding_service,
)


@pytest.fixture(autouse=True)
def _reset_embedding_singleton() -> None:
    EmbeddingService._instance = None
    get_settings.cache_clear()
    yield
    EmbeddingService._instance = None
    get_settings.cache_clear()


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kwargs: (
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        if isinstance(texts, str)
        else np.asarray([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
    )
    return model


def test_looks_like_sentence_transformer_model(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _looks_like_sentence_transformer_model(empty) is False

    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "modules.json").write_text("{}", encoding="utf-8")
    assert _looks_like_sentence_transformer_model(valid) is True


def test_local_model_path_preferred_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dir = tmp_path / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
    local_dir.mkdir(parents=True)
    (local_dir / "modules.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("OFFLINE_MODE", "false")
    monkeypatch.setenv("EMBEDDING_MODEL_LOCAL_PATH", str(local_dir))
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    get_settings.cache_clear()

    fake = _fake_model()
    with patch(
        "app.services.embedding_service.SentenceTransformer",
        return_value=fake,
    ) as ctor:
        service = get_embedding_service()
        loaded = service.load_model()

    assert loaded is fake
    ctor.assert_called_once()
    args, kwargs = ctor.call_args
    assert Path(args[0]) == local_dir.resolve()
    assert kwargs.get("local_files_only") is True
    assert service._model_source == str(local_dir.resolve())


def test_offline_mode_requires_local_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing-model"
    monkeypatch.setenv("OFFLINE_MODE", "true")
    monkeypatch.setenv("EMBEDDING_MODEL_LOCAL_PATH", str(missing))
    get_settings.cache_clear()

    with patch("app.services.embedding_service.SentenceTransformer") as ctor:
        service = get_embedding_service()
        with pytest.raises(RuntimeError, match="Local embedding model is not available"):
            service.load_model()
    ctor.assert_not_called()


def test_offline_mode_loads_local_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dir = tmp_path / "offline-model"
    local_dir.mkdir()
    (local_dir / "config_sentence_transformers.json").write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setenv("OFFLINE_MODE", "true")
    monkeypatch.setenv("EMBEDDING_MODEL_LOCAL_PATH", str(local_dir))
    monkeypatch.setenv("EMBEDDING_MODEL", "should-not-be-used")
    get_settings.cache_clear()

    fake = _fake_model()
    with patch(
        "app.services.embedding_service.SentenceTransformer",
        return_value=fake,
    ) as ctor:
        service = get_embedding_service()
        service.load_model()

    args, kwargs = ctor.call_args
    assert Path(args[0]) == local_dir.resolve()
    assert kwargs.get("local_files_only") is True
    assert "should-not-be-used" not in str(args)


def test_embed_text_and_documents_with_mocked_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dir = tmp_path / "embed-model"
    local_dir.mkdir()
    (local_dir / "modules.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("OFFLINE_MODE", "true")
    monkeypatch.setenv("EMBEDDING_MODEL_LOCAL_PATH", str(local_dir))
    get_settings.cache_clear()

    fake = _fake_model()
    with patch(
        "app.services.embedding_service.SentenceTransformer",
        return_value=fake,
    ):
        service = get_embedding_service()
        single = service.embed_text("hello world")
        batch = service.embed_documents(["one", "two"])

    assert single == pytest.approx([0.1, 0.2, 0.3])
    assert len(batch) == 2
    assert batch[0] == pytest.approx([0.1, 0.2, 0.3])
    assert fake.encode.call_count == 2


def test_online_mode_falls_back_to_remote_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "no-local-model"
    monkeypatch.setenv("OFFLINE_MODE", "false")
    monkeypatch.setenv("EMBEDDING_MODEL_LOCAL_PATH", str(missing))
    monkeypatch.setenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    get_settings.cache_clear()

    fake = _fake_model()
    with patch(
        "app.services.embedding_service.SentenceTransformer",
        return_value=fake,
    ) as ctor:
        service = get_embedding_service()
        service.load_model()

    args, kwargs = ctor.call_args
    assert args[0] == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert kwargs.get("local_files_only") is False
