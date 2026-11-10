"""X-VLA extension for the pinned RLinf submodule (not the legacy root)."""
_registered = False

def register():
    global _registered
    if _registered:
        return
    from rlinf.models import register_model
    register_model('xvla', build_model, category='embodied')
    _registered = True

def build_model(cfg, torch_dtype):
    from .policy import XVLAPolicy
    if cfg.get('is_lora', False):
        raise ValueError('Use xvla.update_scope; RLinf factory LoRA injection must be disabled')
    model = XVLAPolicy.from_checkpoint(cfg, torch_dtype)
    if cfg.get('xvla', {}).get('initial_sft_checkpoint') is not None:
        from .rl_initialization import initialize_from_sft
        initialize_from_sft(model, cfg)
    return model
