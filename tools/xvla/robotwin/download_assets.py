"""Download and verify the three original RoboTwin simulator archives."""
from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download import download_file

REPO = "TianxingChen/RoboTwin2.0"
ARCHIVES = {
    "background_texture.zip": (10970687027, "54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070"),
    "embodiments.zip": (219859313, "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"),
    "objects.zip": (3736741694, "39d94a8a80e45f98a7f2ee936f70ca073a8f63484dcffc82ded2b9fd8f8475df"),
}


def main(dest: Path, revision: str):
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        size, checksum = ARCHIVES[name]
        entry = dict(rfilename=name, size=size, lfs=dict(size=size, sha256=checksum))
        for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
            try:
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
