"""Download and verify the three original RoboTwin simulator archives."""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download import download_file

REPO = "TianxingChen/RoboTwin2.0"
ARCHIVES = {
    "background_texture.zip": (10970687027, "54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070"),
    "embodiments.zip": (219859313, "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"),
    "objects.zip": (3736741694, "39d94a8a80e45f98a7f2ee936f70ca073a8f63484dcffc82ded2b9fd8f8475df"),
}


def finish_tail(partial: Path, size: int, endpoint: str, revision: str) -> None:
    """Fetch a small remaining tail with bounded parallel ranges."""
    import requests

    offset = partial.stat().st_size if partial.is_file() else 0
    if not 0 < size - offset <= 512 * 2**20:
        return

    chunk_size, workers = 2**21, 4
    name = "background_texture.zip"
    url = f"{endpoint}/datasets/{REPO}/resolve/{revision}/{name}"

    def fetch_chunk(start: int) -> bytes:
        end = min(start + chunk_size, size) - 1
        headers = {"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"}
        for attempt in range(3):
            try:
                with requests.get(url, params={"download": "true", "xvla_resume": str(time.time_ns())},
                                  headers=headers, stream=True, timeout=(10, 30)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Unexpected response to archive tail range")
                    data = bytearray()
                    for block in response.iter_content(64 * 1024):
                        data.extend(block)
                        if len(data) > end - start + 1:
                            raise ValueError("Archive tail response exceeds requested range")
                if len(data) != end - start + 1:
                    raise ValueError("Archive tail response is incomplete")
                return bytes(data)
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise

    print(json.dumps(dict(event="asset_tail_parallel", file=name, bytes=offset, total=size, workers=workers)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, partial.open("ab") as stream:
        stride = chunk_size * workers
        for start in range(offset, size, stride):
            for data in pool.map(fetch_chunk, range(start, min(start + stride, size), chunk_size)):
                stream.write(data)
            stream.flush()
            print(json.dumps(dict(event="asset_tail_progress", file=name, bytes=stream.tell(), total=size)), flush=True)


def main(dest: Path, revision: str):
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        size, checksum = ARCHIVES[name]
        entry = dict(rfilename=name, size=size, lfs=dict(size=size, sha256=checksum))
        for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
            try:
                if name == "background_texture.zip" and endpoint == "https://hf-mirror.com":
                    try:
                        finish_tail(dest / f"{name}.http.part", size, endpoint, revision)
                    except (OSError, ValueError) as error:
                        print(json.dumps(dict(event="asset_tail_fallback", file=name, error_type=type(error).__name__)), flush=True)
                # Open-ended streams from the official Xet bridge were much
                # slower on IDAC; bounded requests preserve the same bytes.
                return {**download_file(REPO, revision, entry, dest, endpoint, "dataset", range_bytes=2**23),
                        "endpoint": endpoint, "range_bytes": 2**23}
            except Exception as error:
                print(json.dumps(dict(archive=name, endpoint=endpoint, error_type=type(error).__name__)), flush=True)
        raise RuntimeError(f"Failed to verify/download {name} at {revision}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, ARCHIVES))
    tmp = dest / "verified.json.tmp"
    tmp.write_text(json.dumps(dict(repo_id=REPO, revision=revision, files=files), indent=2) + "\n")
    tmp.replace(dest / "verified.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
