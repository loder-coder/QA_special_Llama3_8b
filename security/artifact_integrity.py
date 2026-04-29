from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACTS_DIR / "manifest.hash"


def _iter_artifact_files(artifacts_dir: Path, manifest_path: Path) -> list[Path]:
    if not artifacts_dir.exists():
        return []
    manifest_resolved = manifest_path.resolve()
    files = []
    for path in artifacts_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == manifest_resolved:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    root = Path(artifacts_dir)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in _iter_artifact_files(root, manifest):
        relative_path = path.relative_to(root).as_posix()
        rows.append(f"{sha256_file(path)}  {relative_path}")
    with manifest.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(rows))
        if rows:
            file.write("\n")
    return manifest


def _read_manifest(manifest_path: Path) -> dict[str, str]:
    expected = {}
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                digest, relative_path = stripped.split(None, 1)
            except ValueError as exc:
                raise ValueError(f"invalid artifact manifest line {line_number}: {manifest_path}") from exc
            expected[relative_path.strip()] = digest.strip()
    return expected


def verify_artifact_manifest(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> None:
    root = Path(artifacts_dir)
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise RuntimeError(f"artifact manifest is missing: {manifest}")

    expected = _read_manifest(manifest)
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _iter_artifact_files(root, manifest)
    }

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(path for path, digest in expected.items() if actual.get(path) != digest)
    if missing or unexpected or mismatched:
        details = {
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
        }
        raise RuntimeError(f"artifact integrity check failed: {details}")
