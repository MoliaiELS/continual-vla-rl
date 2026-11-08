"""Fail before starting Ray for unsupported training algorithms."""

def validate_algorithm(cfg):
    if cfg.runner.only_eval:
        return
    pair=(cfg.algorithm.adv_type,cfg.algorithm.loss_type)
    if pair not in {('gae','actor_critic'),('grpo','actor')}:
        raise NotImplementedError('X-VLA supports PPO (gae,actor_critic) or GRPO (grpo,actor); '
                                  'SAC/IQL/OPD/DAgger and other losses are not implemented')
