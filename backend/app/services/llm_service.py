"""Provider-aware LLM client service (OpenRouter / Foundry Local)."""

from __future__ import annotations

import logging
import threading
from typing import Final

from openai import APIError, APITimeoutError, AsyncOpenAI, OpenAIError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM client is missing required configuration."""


class LLMGenerationError(RuntimeError):
    """Raised when the LLM fails to generate a usable answer."""


class LLMService:
    """Generate chat completions through the configured LLM provider.

    Supports:
    - ``openrouter``: OpenAI-compatible AsyncOpenAI client
    - ``foundry_local``: Microsoft Foundry Local on-device inference

    The service is a singleton so clients/resources are reused across requests.
    """

    _instance: LLMService | None = None
    _lock: Final[threading.RLock] = threading.RLock()

    def __new__(cls) -> LLMService:
        """Return the shared ``LLMService`` instance.

        Returns:
            LLMService: Process-wide singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._client = None
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes only once for the singleton."""
        if getattr(self, "_initialized", False):
            return
        self._client: AsyncOpenAI | None = None
        self._initialized = True

    def _get_client(self) -> AsyncOpenAI:
        """Create or return the shared AsyncOpenAI client for OpenRouter.

        Returns:
            AsyncOpenAI: Configured OpenAI-compatible client.

        Raises:
            LLMConfigurationError: If the API key is missing.
        """
        settings = get_settings()
        api_key = settings.LLM_API_KEY.strip()
        if not api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured. "
                "Set it in the backend .env file to enable LLM answers."
            )

        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=settings.LLM_BASE_URL,
                        timeout=60.0,
                    )
        return self._client

    async def _generate_openrouter_answer(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate an answer through OpenRouter.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            str: Model-generated answer text.

        Raises:
            LLMGenerationError: If generation fails.
        """
        settings = get_settings()
        client = self._get_client()

        logger.info(
            "LLM call started (provider=%s, model=%s).",
            settings.LLM_PROVIDER,
            settings.LLM_MODEL,
        )

        try:
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        except APITimeoutError as exc:
            logger.exception("LLM request timed out.")
            raise LLMGenerationError("LLM request timed out.") from exc
        except APIError as exc:
            logger.exception(
                "LLM API error (status=%s).",
                getattr(exc, "status_code", None),
            )
            raise LLMGenerationError("LLM API request failed.") from exc
        except OpenAIError as exc:
            logger.exception("LLM client error.")
            raise LLMGenerationError("LLM client request failed.") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected LLM failure.")
            raise LLMGenerationError("Unexpected LLM failure.") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMGenerationError(
                "LLM returned an unexpected response shape."
            ) from exc

        answer = (content or "").strip()
        if not answer:
            raise LLMGenerationError("LLM returned an empty answer.")

        logger.info("LLM call completed (provider=%s).", settings.LLM_PROVIDER)
        return answer

    async def _generate_foundry_local_answer(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate an answer through Microsoft Foundry Local.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            str: Model-generated answer text.

        Raises:
            LLMGenerationError: If local inference fails.
            RuntimeError: If Foundry Local lifecycle operations fail.
        """
        from app.services.foundry_local_service import get_foundry_local_service

        settings = get_settings()
        logger.info(
            "LLM call started (provider=%s, model=%s).",
            settings.LLM_PROVIDER,
            settings.FOUNDRY_MODEL_ALIAS,
        )

        try:
            answer = await get_foundry_local_service().generate_answer(messages)
        except ValueError:
            raise
        except RuntimeError:
            # Preserve Foundry lifecycle/inference errors for FastAPI mapping.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected Foundry Local failure.")
            raise LLMGenerationError(
                "Unexpected Foundry Local failure."
            ) from exc

        logger.info("LLM call completed (provider=%s).", settings.LLM_PROVIDER)
        return answer

    async def generate_answer(self, messages: list[dict[str, str]]) -> str:
        """Generate an assistant answer from chat messages.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            str: Model-generated answer text.

        Raises:
            ValueError: If ``messages`` is empty.
            LLMConfigurationError: If the provider is unsupported or misconfigured.
            LLMGenerationError: If generation fails.
            RuntimeError: If Foundry Local lifecycle/inference fails.
        """
        if not messages:
            raise ValueError("messages cannot be empty.")

        settings = get_settings()
        provider = settings.LLM_PROVIDER
        logger.info("LLM provider selected: %s", provider)

        if provider == "openrouter":
            return await self._generate_openrouter_answer(messages)

        if provider == "foundry_local":
            return await self._generate_foundry_local_answer(messages)

        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Valid values: openrouter, foundry_local."
        )


def get_llm_service() -> LLMService:
    """Return the shared ``LLMService`` singleton.

    Returns:
        LLMService: Process-wide LLM service instance.
    """
    return LLMService()
