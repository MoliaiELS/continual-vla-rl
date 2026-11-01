"""Create a checkout-specific LIBERO configuration without interactive imports."""
import json
import os
from pathlib import Path

import yaml

repo = Path(os.environ["FYP_REPO"]).resolve()
data = Path(os.environ["FYP_DATA_ROOT"])
root = repo / "LIBERO/libero/libero"
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "assets": str(root / "assets"),
    "datasets": str(data / "datasets"),
}
dest = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(yaml.safe_dump(config), encoding="utf-8")
print(json.dumps(config, indent=2))
