from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .hashing import sha256_file
from .io import atomic_write, ensure_dir, read_json, write_json


def _filename_from_url(url: str) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    if not name:
        raise ValueError(f"Cannot determine filename from URL: {url}")
    return name


def load_checksum_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        obj = read_json(path)
        if not isinstance(obj, dict):
            raise ValueError(f"Checksum manifest must be a dict: {path}")
        return obj
    return {"version": 1, "files": {}}


def save_checksum_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    write_json(path, manifest)


def download_with_verification(
    url: str,
    dst_dir: Path,
    *,
    checksum_manifest_path: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
    timeout_sec: int = 60,
    retries: int = 3,
) -> Path:
    """
    Download `url` to `dst_dir/<filename>` and verify SHA-256 when possible:
    - If `expected_sha256` is provided, it is enforced.
    - Else if `checksum_manifest_path` contains a checksum for this filename, it is enforced.
    - Else the file's SHA-256 is computed and written into the manifest for future verification.
    """
    ensure_dir(dst_dir)
    filename = _filename_from_url(url)
    dst = dst_dir / filename
    tmp = dst.with_suffix(dst.suffix + ".part")

    manifest: Optional[Dict[str, Any]] = None
    if checksum_manifest_path is not None:
        manifest = load_checksum_manifest(checksum_manifest_path)

    manifest_sha = None
    if manifest is not None:
        manifest_sha = manifest.get("files", {}).get(filename, {}).get("sha256")

    enforced_sha = expected_sha256 or manifest_sha

    if dst.exists() and enforced_sha:
        got = sha256_file(dst)
        if got.lower() != enforced_sha.lower():
            raise ValueError(f"Checksum mismatch for {dst}: expected {enforced_sha}, got {got}")
        return dst

    if dst.exists() and not enforced_sha and manifest is not None:
        got = sha256_file(dst)
        manifest.setdefault("files", {}).setdefault(filename, {})["sha256"] = got
        manifest["files"][filename]["url"] = url
        save_checksum_manifest(checksum_manifest_path, manifest)
        return dst

    req = Request(url, headers={"User-Agent": "msmarco-bnr/0.1"})
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout_sec) as r:
                total = r.headers.get("Content-Length")
                # Stream download to .part then atomically rename.
                ensure_dir(tmp.parent)
                with tmp.open("wb") as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(dst)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to download {url}") from last_err

    got = sha256_file(dst)
    if enforced_sha and got.lower() != enforced_sha.lower():
        raise ValueError(f"Checksum mismatch for {dst}: expected {enforced_sha}, got {got}")

    if manifest is not None:
        manifest.setdefault("files", {}).setdefault(filename, {})["sha256"] = got
        manifest["files"][filename]["url"] = url
        save_checksum_manifest(checksum_manifest_path, manifest)

    return dst

