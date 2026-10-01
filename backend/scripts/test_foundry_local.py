"""Smoke-test Microsoft Foundry Local on this Windows machine.

This script verifies that Foundry Local can initialize, list models,
download/load a lightweight model, run a short chat completion, and unload
cleanly — before any RAG / OpenRouter integration work.

No API keys are required. Foundry Local runs entirely on-device.
"""

from __future__ import annotations

import sys
from typing import Any

PREFERRED_ALIAS = "qwen2.5-0.5b"
TEST_PROMPT = "Reply with exactly: Foundry Local connection successful"


def _print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _model_label(model: Any) -> str:
    alias = getattr(model, "alias", None) or "?"
    model_id = getattr(model, "id", None) or "?"
    cached = getattr(model, "is_cached", None)
    loaded = getattr(model, "is_loaded", None)
    flags: list[str] = []
    if cached is True:
        flags.append("cached")
    if loaded is True:
        flags.append("loaded")
    suffix = f" [{', '.join(flags)}]" if flags else ""
    return f"{alias}  (id={model_id}){suffix}"


def _list_models(catalog: Any) -> list[Any]:
    models = list(catalog.list_models() or [])
    if not models:
        return []

    print(f"Found {len(models)} model(s) in the Foundry Local catalog:\n")
    for index, model in enumerate(models, start=1):
        print(f"  {index:2d}. {_model_label(model)}")

    try:
        cached = list(catalog.get_cached_models() or [])
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"\nNote: could not list cached models ({exc}).")
        cached = []

    if cached:
        print(f"\nCached locally ({len(cached)}):")
        for model in cached:
            print(f"  - {_model_label(model)}")

    return models


def _resolve_preferred_model(catalog: Any, models: list[Any]) -> Any | None:
    aliases = {
        str(getattr(model, "alias", "") or "").strip().lower(): model
        for model in models
    }
    preferred = aliases.get(PREFERRED_ALIAS.lower())
    if preferred is not None:
        return preferred

    try:
        model = catalog.get_model(PREFERRED_ALIAS)
    except Exception:
        model = None

    if model is not None:
        return model

    print()
    print(f"Preferred model '{PREFERRED_ALIAS}' was not found in the catalog.")
    print("Available aliases:")
    for model in models:
        alias = getattr(model, "alias", None)
        if alias:
            print(f"  - {alias}")
    return None


def _download_with_progress(model: Any) -> None:
    if getattr(model, "is_cached", False):
        print("Model is already cached. Skipping download.")
        return

    print("Downloading model (this may take a while)...")

    def on_progress(percent: float) -> None:
        bar_width = 28
        filled = int(bar_width * max(0.0, min(100.0, percent)) / 100.0)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\r  [{bar}] {percent:5.1f}%", end="", flush=True)

    model.download(progress_callback=on_progress)
    print()
    print("Download complete.")


def _extract_response_text(response: Any) -> str:
    try:
        return str(response.choices[0].message.content)
    except Exception:
        pass

    if isinstance(response, str):
        return response

    return repr(response)


def main() -> int:
    _print_header("Foundry Local connectivity test")
    print("App name : corporate_document_assistant")
    print("Preferred: " + PREFERRED_ALIAS)
    print("No cloud API keys are used.\n")

    model = None

    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ImportError as exc:
        print("ERROR: foundry_local_sdk is not installed in this environment.")
        print("Install with:")
        print("  pip install foundry-local-sdk")
        print("or on Windows with WinML:")
        print("  pip install foundry-local-sdk-winml")
        print(f"Details: {exc}")
        return 1

    try:
        print("[1/8] Initializing Foundry Local...")
        config = Configuration(app_name="corporate_document_assistant")
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance
        print("Initialization OK.")

        print("\n[2/8] Listing available catalog models...")
        catalog = manager.catalog
        models = _list_models(catalog)
        if not models:
            print("No models were returned by catalog.list_models().")
            print("Ensure Foundry Local / native binaries are installed correctly.")
            return 1

        print("\n[3/8] Selecting lightweight test model...")
        model = _resolve_preferred_model(catalog, models)
        if model is None:
            print("\nExiting gracefully because the preferred model is unavailable.")
            return 1
        print(f"Selected: {_model_label(model)}")

        print("\n[4/8] Ensuring model is downloaded...")
        _download_with_progress(model)

        print("\n[5/8] Loading model into memory...")
        model.load()
        print("Model loaded.")

        print("\n[6/8] Sending local test prompt...")
        print(f"Prompt: {TEST_PROMPT}")
        client = model.get_chat_client()
        try:
            client.settings.temperature = 0.0
            client.settings.max_tokens = 64
        except Exception:
            # Settings APIs may vary slightly across SDK builds.
            pass

        response = client.complete_chat(
            [
                {"role": "user", "content": TEST_PROMPT},
            ]
        )
        text = _extract_response_text(response).strip()

        print("\n[7/8] Model response:")
        print("-" * 60)
        print(text if text else "(empty response)")
        print("-" * 60)

        print("\n[8/8] Unloading model...")
        model.unload()
        model = None
        print("Model unloaded cleanly.")

        _print_header("SUCCESS")
        print("Foundry Local is working on this machine.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level diagnostic script
        print("\nERROR: Foundry Local test failed.")
        print(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if model is not None:
            try:
                print("\nCleaning up: unloading model...")
                model.unload()
                print("Cleanup complete.")
            except Exception as cleanup_exc:  # noqa: BLE001
                print(f"Cleanup warning: could not unload model ({cleanup_exc}).")


if __name__ == "__main__":
    sys.exit(main())
