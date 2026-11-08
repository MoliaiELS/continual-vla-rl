"""Apply the audited, opt-in X-VLA patch to the exact pinned RLinf runtime.

The checkout must already be the pinned submodule commit. Unrelated local edits
are preserved; edits to a patched file are refused instead of reset or overwritten.
No network, environment installation or GPU work is performed by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PINNED_RLINF = "1bf78980d9339cbc67497eddceb9a1e3a74e2acc"


def _git(runtime: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(runtime), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _digest(path: Path) -> str:
    # Git's Windows checkout may use CRLF. The patch manifest hashes canonical
    # Git source bytes, so line endings are normalized and nothing else is changed.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def apply_runtime_patch(
    runtime: Path,
    *,
    check_only: bool = False,
    patch: Path | None = None,
    manifest: Path | None = None,
) -> dict:
    runtime = runtime.resolve()
    repository = Path(__file__).resolve().parents[2]
    patch = (patch or repository / "patches/rlinf-xvla.patch").resolve()
    manifest = (manifest or repository / "patches/rlinf-xvla.json").resolve()
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    expected = metadata["upstream_commit"]
    if _digest(patch) != metadata["patch_sha256"]:
        raise RuntimeError("Patch checksum does not match its audited manifest")
    # Alternative manifests are useful for unit fixtures, but the normal CLI
    # always uses the audited pinned version above.
    if _git(runtime, "rev-parse", "HEAD") != expected:
        raise RuntimeError(f"RLinf HEAD must be {expected}; refusing to patch another version")
    before, after, changed = [], [], []
    for item in metadata["files"]:
        target = (runtime / item["path"]).resolve()
        if not target.is_relative_to(runtime):
            raise ValueError("Patch manifest path escapes runtime checkout")
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = _digest(target)
        if actual == item["before_sha256"]:
            before.append(item["path"])
        elif actual == item["after_sha256"]:
            after.append(item["path"])
        else:
            changed.append(item["path"])
    if changed or (before and after):
        raise RuntimeError(
            "Patched files contain user changes or a partial patch; preserved unchanged: "
            + ", ".join(changed or before + after)
        )
    if after:
        return {"status": "already_applied", "upstream_commit": expected, "files": len(after)}
    _git(runtime, "apply", "--check", "--whitespace=error-all", str(patch))
    if check_only:
        return {"status": "applicable", "upstream_commit": expected, "files": len(before)}
    _git(runtime, "apply", "--whitespace=error-all", str(patch))
    for item in metadata["files"]:
        if _digest(runtime / item["path"]) != item["after_sha256"]:
            raise RuntimeError(f"Post-apply checksum mismatch: {item['path']}")
    return {"status": "applied", "upstream_commit": expected, "files": len(before)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parents[2] / "third_party/rlinf")
    parser.add_argument("--check", action="store_true", help="Verify applicability without modifying files")
    args = parser.parse_args()
    metadata = json.loads((Path(__file__).resolve().parents[2] / "patches/rlinf-xvla.json").read_text())
    if metadata["upstream_commit"] != PINNED_RLINF:
        raise RuntimeError("Unexpected runtime manifest revision")
    print(json.dumps(apply_runtime_patch(args.runtime, check_only=args.check), sort_keys=True))


if __name__ == "__main__":
    main()
