"""Record the actual GPU, inputs, dependency identities and immutable commits."""
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit(path):
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD"], check=True)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def receipt(path, expected):
    obj = json.loads((path / "verified.json").read_text())
    if obj["revision"] != expected or not obj["files"]:
        raise ValueError(f"Unverified snapshot: {path}")
    for item in obj["files"]:
        file = path / item["file"]
        if file.stat().st_size != item["bytes"]:
            raise ValueError(f"File size changed after verification: {file}")
        if file.suffix == ".json" and digest(file) != item["sha256"]:
            raise ValueError(f"File checksum changed after verification: {file}")
    return {"path": str(path), **obj}


def main(mode):
    import torch
    import sapien
    import curobo
    import mplib
    import transformers

    if torch.cuda.device_count() != 1:
        raise ValueError("RoboTwin jobs require exactly one Slurm assigned CUDA GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid).lower().removeprefix("gpu-")
    data = Path(os.environ["FYP_ROBOTWIN_DATA"])
    out = Path(os.environ["FYP_ROBOTWIN_JOB"])
    (out / "allocated_gpu_uuids.json").write_text(json.dumps([uuid]) + "\n")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,driver_version", "--format=csv,noheader"], text=True)
    ((_, pci, driver),) = (tuple(map(str.strip, row)) for row in csv.reader(query.splitlines())
                          if row[0].strip().lower().removeprefix("gpu-") == uuid)
    match = re.fullmatch(r"[\da-fA-F]+:([\da-fA-F]{2}):([\da-fA-F]{2})\.([\da-fA-F])", pci)
    if not match:
        raise ValueError(f"Cannot identify Slurm GPU's PCI address: {pci}")
    # SAPIEN 3's parser accepts the hexadecimal PCI bus:device.function form.
    device = f"pci:{match[1]}:{match[2]}.{match[3]}"
    (out / "vulkan-device.txt").write_text(device + "\n")
    repo = Path(__file__).resolve().parents[3]
    rev = dict(line.split("=", 1) for line in Path(__file__).with_name("runtime.env").read_text().splitlines())
    robot = Path(os.environ["FYP_ROBOTWIN_ROOT"])
    code_paths = {name: path for name, path in (
        ("repository", repo), ("RLINF", repo / "third_party/rlinf"), ("ROBOTWIN", robot),
        ("XVLA", Path(os.environ["FYP_ROBOTWIN_XVLA"])), ("CUROBO", Path(os.environ["FYP_ROBOTWIN_CUROBO"])),
    )}
    code = {name: {"path": str(path), "commit": commit(path)} for name, path in code_paths.items()}
    for name in ("RLINF", "ROBOTWIN", "XVLA", "CUROBO"):
        if code[name]["commit"] != rev[f"{name}_REV"]:
            raise ValueError(f"Wrong pinned {name} checkout: {code[name]}")
    files = sorted((robot / "task_config").glob("*.yml"))
    manifest = dict(job_id=os.environ["SLURM_JOB_ID"], hostname=os.uname().nodename, mode=mode, code=code,
                    python=sys.version, prefix=sys.prefix, cuda=torch.version.cuda,
                    environment={name: metadata.version(name) for name in (
                        "torch", "torchvision", "transformers", "sapien", "mplib", "nvidia-curobo", "warp-lang", "numpy", "scipy"
                    )}, imports={m.__name__: m.__file__ for m in (torch, sapien, curobo, mplib, transformers)},
                    gpu=dict(uuid=uuid, model=properties.name, pci=pci, driver=driver, vulkan_device=device,
                             slurm_visible=os.environ["CUDA_VISIBLE_DEVICES"], total_bytes=properties.total_memory),
                    camera=dict(sensor="D435", near_m=0.1, far_m=100, shader="rt", samples=32),
                    configs={str(f.relative_to(robot)): digest(f) for f in files},
                    pip_freeze_sha256=digest(data / "evidence/requirements.freeze.txt"),
                    install_patches=json.loads((data / "evidence/install-patches.json").read_text()))
    if mode != "probe":
        assets = data / "sim-assets" / rev["ASSETS_REV"]
        if not (assets / ".extracted").is_file():
            raise ValueError("Pinned simulator assets have not finished extracting")
        embodiment = robot / "assets/embodiments/aloha-agilex"
        manifest.update(assets=receipt(data / "archives" / rev["ASSETS_REV"], rev["ASSETS_REV"]),
                        embodiment={f.name: digest(f) for f in sorted(embodiment.glob("*.yml"))},
                        model=receipt(Path(os.environ["FYP_ROBOTWIN_MODEL"]), rev["MODEL_REV"]))
    (out / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"ROBOTWIN_RUNTIME {out / 'runtime.json'} {device}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
