"""Microsoft Foundry Local model lifecycle and inference service."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Final

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class FoundryLocalService:
    """Manage Foundry Local model download/load lifecycle and chat inference.

    The service is a process-wide singleton. Models are downloaded once and kept
    loaded across requests for performance. Call ``unload_model()`` explicitly
    when shutdown cleanup is required.
    """

    _instance: FoundryLocalService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> FoundryLocalService:
        """Return the shared ``FoundryLocalService`` instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._manager = None
                    instance._model = None
                    instance._model_alias = None
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._manager: Any | None = None
        self._model: Any | None = None
        self._model_alias: str | None = None
        self._initialized = True

    def initialize(self) -> None:
        """Initialize the Foundry Local manager singleton.

        Raises:
            RuntimeError: If Foundry Local SDK initialization fails.
        """
        if self._manager is not None:
            return

        with self._lock:
            if self._manager is not None:
                return

            try:
                from foundry_local_sdk import Configuration, FoundryLocalManager
            except ImportError as exc:
                raise RuntimeError(
                    "Foundry Local SDK is not installed. "
                    "Install foundry-local-sdk-winml for Windows."
                ) from exc

            settings = get_settings()
            logger.info(
                "Foundry Local initialization started (app_name=%s).",
                "corporate_document_assistant",
            )
            try:
                config = Configuration(app_name="corporate_document_assistant")
                FoundryLocalManager.initialize(config)
                self._manager = FoundryLocalManager.instance
            except Exception as exc:  # noqa: BLE001
                logger.exception("Foundry Local initialization failed.")
                raise RuntimeError(
                    "Failed to initialize Microsoft Foundry Local."
                ) from exc

            logger.info(
                "Foundry Local initialization complete (provider=%s).",
                settings.LLM_PROVIDER,
            )

    def ensure_model(self) -> Any:
        """Resolve the configured model and download it when not cached.

        Returns:
            Any: Foundry Local model handle.

        Raises:
            RuntimeError: If the model cannot be found or downloaded.
        """
        self.initialize()
        assert self._manager is not None

        settings = get_settings()
        alias = settings.FOUNDRY_MODEL_ALIAS.strip() or "qwen2.5-0.5b"
        logger.info("Foundry Local model selection (alias=%s).", alias)

        with self._lock:
            if (
                self._model is not None
                and self._model_alias == alias
                and getattr(self._model, "is_cached", False)
            ):
                return self._model

            try:
                model = self._manager.catalog.get_model(alias)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Foundry Local model lookup failed.")
                raise RuntimeError(
                    f"Foundry Local model '{alias}' was not found in the catalog."
                ) from exc

            if model is None:
                raise RuntimeError(
                    f"Foundry Local model '{alias}' was not found in the catalog."
                )

            try:
                if not getattr(model, "is_cached", False):
                    logger.info(
                        "Foundry Local model is not cached; downloading (alias=%s).",
                        alias,
                    )
                    model.download()
                    logger.info(
                        "Foundry Local model download complete (alias=%s).",
                        alias,
                    )
                else:
                    logger.info(
                        "Foundry Local model already cached (alias=%s).",
                        alias,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Foundry Local model download failed.")
                raise RuntimeError(
                    f"Failed to download Foundry Local model '{alias}'."
                ) from exc

            self._model = model
            self._model_alias = alias
            return model

    def load_model(self) -> Any:
        """Load the configured model into memory if it is not already loaded.

        Returns:
            Any: Loaded Foundry Local model handle.

        Raises:
            RuntimeError: If model loading fails.
        """
        model = self.ensure_model()
        alias = self._model_alias or get_settings().FOUNDRY_MODEL_ALIAS

        with self._lock:
            try:
                if getattr(model, "is_loaded", False):
                    logger.info(
                        "Foundry Local model already loaded (alias=%s).",
                        alias,
                    )
                    return model

                logger.info("Foundry Local model loading started (alias=%s).", alias)
                model.load()
                logger.info(
                    "Foundry Local model loading complete (alias=%s).",
                    alias,
                )
                return model
            except Exception as exc:  # noqa: BLE001
                logger.exception("Foundry Local model load failed.")
                raise RuntimeError(
                    f"Failed to load Foundry Local model '{alias}'."
                ) from exc

    def _generate_answer_sync(self, messages: list[dict[str, str]]) -> str:
        """Run blocking Foundry Local chat completion.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            str: Generated answer text.

        Raises:
            RuntimeError: If inference fails or returns an empty answer.
        """
        model = self.load_model()
        alias = self._model_alias or get_settings().FOUNDRY_MODEL_ALIAS
        settings = get_settings()

        logger.info("Foundry Local inference started (alias=%s).", alias)
        try:
            client = model.get_chat_client()
            try:
                client.settings.temperature = settings.LLM_TEMPERATURE
                client.settings.max_tokens = settings.LLM_MAX_TOKENS
            except Exception:
                # Optional settings surface may differ across SDK builds.
                pass

            response = client.complete_chat(messages)
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    "Foundry Local returned an unexpected response shape."
                ) from exc

            answer = (content or "").strip()
            if not answer:
                raise RuntimeError("Foundry Local returned an empty answer.")
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Foundry Local inference failed.")
            raise RuntimeError(
                f"Foundry Local inference failed for model '{alias}'."
            ) from exc

        logger.info("Foundry Local inference complete (alias=%s).", alias)
        return answer

    async def generate_answer(self, messages: list[dict[str, str]]) -> str:
        """Generate an answer using the local Foundry model.

        Args:
            messages: OpenAI-format chat messages (system/user/assistant).

        Returns:
            str: Model-generated answer text.

        Raises:
            ValueError: If ``messages`` is empty.
            RuntimeError: If local inference fails.
        """
        if not messages:
            raise ValueError("messages cannot be empty.")

        return await asyncio.to_thread(self._generate_answer_sync, messages)

    def unload_model(self) -> None:
        """Unload the currently loaded Foundry Local model, if any."""
        with self._lock:
            if self._model is None:
                return

            alias = self._model_alias or "unknown"
            try:
                if getattr(self._model, "is_loaded", False):
                    logger.info(
                        "Foundry Local model unload started (alias=%s).",
                        alias,
                    )
                    self._model.unload()
                    logger.info(
                        "Foundry Local model unload complete (alias=%s).",
                        alias,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Foundry Local model unload failed (alias=%s).",
                    alias,
                )
                raise RuntimeError(
                    f"Failed to unload Foundry Local model '{alias}'."
                ) from exc
            finally:
                self._model = None
                self._model_alias = None


def get_foundry_local_service() -> FoundryLocalService:
    """Return the shared ``FoundryLocalService`` singleton.

    Returns:
        FoundryLocalService: Process-wide Foundry Local service instance.
    """
    return FoundryLocalService()
