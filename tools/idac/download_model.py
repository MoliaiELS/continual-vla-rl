"""Download the fixed author SFT snapshot into personal storage and verify it."""
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download


def main():
    data = Path(os.environ["FYP_DATA_ROOT"])
    dest = data / "models/Openvla-oft-SFT-libero-spatial-traj1"
    reports = data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    repo = os.environ["FYP_MODEL_REPOSITORY"]
    revision = os.environ["FYP_MODEL_REVISION"]
    endpoint = os.environ["HF_ENDPOINT"]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    api = HfApi(endpoint=endpoint)
    info = api.model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    files = []
    for entry in info.siblings:
        lfs = entry.lfs
        files.append({"name": entry.rfilename, "size": entry.size,
                      "sha256": lfs.sha256 if lfs else None})
    manifest = {"repository": repo, "revision": revision, "endpoint": endpoint,
                "files": files, "verified": False, "started_at": time.time()}
    manifest_path = reports / "model.snapshot.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["size"] or 0 for f in files)
    print(f"Downloading fixed snapshot {repo}@{revision}: {total} bytes", flush=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=dest,
                      endpoint=endpoint, max_workers=4)
    for entry in files:
        path = dest / entry["name"]
        assert path.is_file(), f"Missing {path}"
        assert path.stat().st_size == entry["size"], f"Wrong size: {path}"
        if entry["sha256"]:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == entry["sha256"], f"SHA256 mismatch: {path}"
        print(f"VERIFIED {entry['name']} {entry['size']}", flush=True)
    index = json.loads((dest / "model.safetensors.index.json").read_text())
    assert all((dest / name).is_file() for name in set(index["weight_map"].values()))
    stats = json.loads((dest / "dataset_statistics.json").read_text())
    assert "libero_spatial_no_noops" in stats
    manifest.update(verified=True, verified_at=time.time(), total_bytes=total,
                    note="LFS files verified by SHA256; all files checked for exact size.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("FYP_MODEL_VERIFIED", revision, flush=True)


if __name__ == "__main__":
    main()
