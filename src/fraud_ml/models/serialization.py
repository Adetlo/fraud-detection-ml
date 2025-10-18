"""joblib persistence with an integrity check (``docs/SECURITY.md``).

joblib/pickle files execute code on load, so a model artifact is a *code* trust boundary, not
a data file. The registry only loads artifacts whose ``metadata.json`` records the SHA-256 of
the joblib file and whose on-disk hash matches; anything else is refused before unpickling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib

from fraud_ml.exceptions import ArtifactIntegrityError

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_model(model: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return sha256_path(path)


def load_model(path: Path, expected_sha256: str) -> Any:
    if not path.is_file():
        raise ArtifactIntegrityError(f"model file missing: {path.name}")
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise ArtifactIntegrityError(
            f"model file hash mismatch for {path.name}: refusing to unpickle an unverified artifact"
        )
    return joblib.load(path)  # trusted: hash verified against metadata written at training time
