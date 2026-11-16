"""Bind the legacy SAPIEN 3.0.0b1 Engine's render system to the Slurm GPU."""
from __future__ import annotations

import os


def bind_renderer():
    alias = os.environ.get("FYP_ROBOTWIN_VULKAN_DEVICE")
    if not alias:
        if os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Cannot use a default Vulkan device under Slurm")
        return
    import sapien
    import sapien.core as core

    if getattr(core.Engine.create_scene, "_robotwin_device", None) == alias:
        return
    device = sapien.Device(alias)

    def create_scene(self, config=None):
        sapien.physx.set_scene_config(config if config is not None else core.SceneConfig())
        return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)])

    create_scene._robotwin_device = alias
    core.Engine.create_scene = create_scene


def probe():
    import numpy as np
    import sapien.core as core
    import torch
    from curobo.wrap.reacher.motion_gen import MotionGenConfig
    from curobo.curobolib import geom_cu

    bind_renderer()
    engine = core.Engine()
    engine.set_renderer(core.SapienRenderer())
    scene = engine.create_scene(core.SceneConfig())
    camera = scene.add_camera("probe", 32, 24, 1.0, 0.1, 100)
    scene.step()
    scene.update_render()
    camera.take_picture()
    color = camera.get_picture("Color")
    if color.shape != (24, 32, 4) or not np.isfinite(color).all():
        raise ValueError(f"Invalid SAPIEN Vulkan frame: {color.shape}")
    assert torch.empty(1, device="cuda:0").is_cuda
    print(f"ROBOTWIN_RENDER_PROBE shape={color.shape} extension={geom_cu.__file__}", flush=True)
