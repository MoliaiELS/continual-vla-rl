"""Apply RoboTwin's documented install-time edits only in the dedicated prefix."""
from importlib.util import find_spec
from pathlib import Path
import hashlib
import json
import os
import sys


def amend(module, relative, before, after):
    path = Path(find_spec(module).origin).parent / relative
    if not path.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"Refusing to patch outside the benchmark prefix: {path}")
    text = path.read_text(encoding="utf-8")
    if before in text:
        path.write_text(text.replace(before, after), encoding="utf-8")
    if after not in path.read_text(encoding="utf-8"):
        raise ValueError(f"Unexpected {module} source: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    if Path(sys.prefix).resolve() != Path(os.environ["FYP_ROBOTWIN_ENV"]).resolve():
        raise ValueError("Wrong RoboTwin Python environment")
    patches = [
        amend("sapien", "wrapper/urdf_loader.py", 'open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
        amend("mplib", "planner.py", "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
              "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
    ]
    target = Path(os.environ["FYP_ROBOTWIN_DATA"]) / "evidence/install-patches.json"
    target.write_text(json.dumps(patches, indent=2) + "\n")


if __name__ == "__main__":
    main()
