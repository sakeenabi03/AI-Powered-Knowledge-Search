"""Download the Sentence Transformers embedding model for offline use.

Run this script once while the machine has internet access. The saved model
directory can then be used by EmbeddingService with OFFLINE_MODE=true.
"""

from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RELATIVE_DIR = Path("models") / "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> int:
    backend_root = Path(__file__).resolve().parents[1]
    destination = (backend_root / DEFAULT_RELATIVE_DIR).resolve()

    print("=" * 60)
    print("Embedding model download (one-time online preparation)")
    print("=" * 60)
    print(f"Source model : {DEFAULT_MODEL_NAME}")
    print(f"Destination  : {destination}")
    print()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print("ERROR: sentence-transformers is not installed in this environment.")
        print("Activate the backend venv and install requirements first.")
        print(f"Details: {exc}")
        return 1

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading / loading model (this may take a few minutes)...")
        model = SentenceTransformer(DEFAULT_MODEL_NAME)
        print("Saving complete local Sentence Transformers model...")
        model.save(str(destination))
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Failed to download or save the embedding model.")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    markers = [
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
    ]
    present = [name for name in markers if (destination / name).exists()]
    print()
    print("Save complete.")
    print(f"Marker files found: {', '.join(present) if present else '(none)'}")
    print()
    print("SUCCESS")
    print("Local embedding model is ready for offline use.")
    print("Set in backend/.env:")
    print("  OFFLINE_MODE=true")
    print(
        "  EMBEDDING_MODEL_LOCAL_PATH="
        "models/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
