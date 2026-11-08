"""Fixed snapshots, resumable HTTP and streamed hashes; never executes HF code."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import quote

# Set before importing huggingface_hub: the native Xet backend stalled here.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

SPECS = {
    "libero": ("2toINF/X-VLA-Libero", "129e71460678b7236cee6fc9707f09d9fa0c3590"),
    "pt": ("2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b"),
}
ALLOWED_SUFFIXES = (".json", ".safetensors", ".bin", ".model", ".txt", ".merges", ".vocab")


def _log(event, **fields):
    # Signed redirect URLs, credentials and exception request headers are secret.
    print(json.dumps({"event": event, **fields}), flush=True)


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def expected_metadata(entry):
    lfs = _value(entry, "lfs")
    size = _value(lfs, "size", _value(entry, "size")) if lfs else _value(entry, "size")
    expected_sha = _value(lfs, "sha256") if lfs else None
    git_blob = _value(entry, "blob_id") if not lfs else None
    if not isinstance(size, int) or size < 0:
        raise ValueError("Remote metadata must declare an exact file size")
    if expected_sha and not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("Invalid remote LFS hash")
    if not expected_sha and not (git_blob and re.fullmatch(r"[0-9a-f]{40}", git_blob)):
        raise ValueError("File has neither LFS SHA256 nor a Git blob checksum")
    return size, expected_sha, git_blob


def verify_file(path, entry):
    size, expected_sha, git_blob = expected_metadata(entry)
    if not path.is_file() or path.stat().st_size != size:
        return None
    sha, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
    actual = sha.hexdigest()
    if expected_sha and actual != expected_sha:
        return None
    if git_blob and blob.hexdigest() != git_blob:
        return None
    return {"file": _value(entry, "rfilename"), "bytes": size, "sha256": actual,
            "lfs_sha256": expected_sha, "git_blob_sha1": git_blob}


def download_file(repo, revision, entry, dest, endpoint):
    import requests

    filename = _value(entry, "rfilename")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid remote asset path")
    path = dest / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(dest.resolve()):
        raise ValueError("Asset path escapes the private snapshot directory")
    record = verify_file(path, entry)
    if record is not None:
        _log("asset_verified_existing", file=filename, bytes=record["bytes"])
        return record
    size, expected_sha, _ = expected_metadata(entry)
    partial = path.with_name(path.name + ".http.part")
    if not partial.exists() and expected_sha:
        # An old Xet partial is untrusted until the final full checksum passes.
        # Preserve it; copy rather than move/delete its existing cache entry.
        cached = sorted(dest.rglob(f"*.{expected_sha}.incomplete"))
        if len(cached) == 1 and cached[0].stat().st_size < size:
            shutil.copyfile(cached[0], partial)
            _log("resume_prefix_copied", file=filename, bytes=partial.stat().st_size)
    if partial.exists() and partial.stat().st_size > size:
        raise ValueError("Resume file exceeds metadata size; manual audit required")
    url = f"{endpoint.rstrip('/')}/{repo}/resolve/{revision}/{quote(filename, safe='/')}"
    failures, hash_restarts, last_report = 0, 0, 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            verified = verify_file(partial, entry)
            if verified is not None:
                partial.replace(path)
                _log("asset_verified", file=filename, bytes=size)
                return verified
            # Xet can leave holes/non-prefix chunks. Never bless a bad prefix.
            if hash_restarts:
                raise ValueError("Completed asset failed its pinned full-file hash twice")
            hash_restarts += 1
            with partial.open("wb"):
                pass
            _log("full_hash_failed_restart_zero", file=filename)
            continue
        before = offset
        try:
            headers = {"Accept-Encoding": "identity", "Range": f"bytes={offset}-"}
            # Refresh expiring signed redirects; the immutable resolve path and
            # final checksum never change. Do not print response.url.
            params = {"download": "true", "xvla_resume": str(time.time_ns())}
            with requests.get(url, headers=headers, params=params, stream=True,
                              timeout=(10, 45)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != size:
                        raise ValueError("Mismatched HTTP resume Content-Range")
                    mode = "ab"
                elif response.status_code == 200:
                    # Range ignored: safely restart using this complete body.
                    offset, mode = 0, "wb"
                else:
                    raise ValueError("Unexpected successful HTTP status")
                with partial.open(mode) as stream:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        if offset + len(block) > size:
                            raise ValueError("HTTP body exceeds pinned asset size")
                        stream.write(block)
                        offset += len(block)
                        if offset - last_report >= 32 * 1024 * 1024:
                            _log("asset_progress", file=filename, bytes=offset, total=size)
                            last_report = offset
            failures = 0 if offset > before else failures + 1
            if failures >= 8:
                raise RuntimeError("HTTP server repeatedly returned no progress")
        except ValueError:
            raise
        except Exception as error:
            after = partial.stat().st_size if partial.exists() else 0
            failures = 0 if after > before else failures + 1
            _log("http_retry", file=filename, bytes=after, error_type=type(error).__name__, failures=failures)
            if failures >= 8:
                raise RuntimeError("HTTP download exhausted bounded retries") from None
            time.sleep(min(2 ** failures, 8))


def main():
    from huggingface_hub import HfApi

    root, name = Path(sys.argv[1]), sys.argv[2]
    repo, revision = SPECS[name]
    dest = root / (repo.split("/")[-1] + "--" + revision[:12])
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
        try:
            info = HfApi(endpoint=endpoint).model_info(repo, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError("Metadata revision differs from requested immutable snapshot")
            records = []
            for entry in info.siblings:
                filename = entry.rfilename
                if not filename.endswith(ALLOWED_SUFFIXES) or filename.endswith(".py"):
                    continue
                records.append(download_file(repo, revision, entry, dest, endpoint))
            if not records or not any(record["file"].endswith((".safetensors", ".bin")) for record in records):
                raise ValueError("No model weight assets verified")
            receipt = {"repo_id": repo, "revision": revision, "endpoint": endpoint,
                       "transfer": "http_range_v1", "hf_hub_disable_xet": True, "files": records}
            temporary = dest / "verified.json.tmp"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(dest / "verified.json")
            _log("snapshot_verified", directory=str(dest), files=len(records))
            return
        except Exception as error:
            # requests exceptions can embed signed query secrets; log type only.
            errors.append({"endpoint": endpoint, "error_type": type(error).__name__})
            _log("endpoint_failed", **errors[-1])
    raise RuntimeError("Fixed snapshot download failed: " + json.dumps(errors))


if __name__ == "__main__":
    main()
