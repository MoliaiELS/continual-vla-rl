# Pinned RLinf X-VLA compatibility patch

`rlinf-xvla.patch` applies only to `RLinf/RLinf@1bf78980d9339cbc67497eddceb9a1e3a74e2acc`.
The pinned submodule is preserved; this patch is tracked separately rather than
claiming another upstream revision. The manifest records the patch SHA256 and
canonical-LF before/after SHA256 of every affected file.

```bash
python tools/xvla/apply_runtime_patch.py --runtime third_party/rlinf --check
python tools/xvla/apply_runtime_patch.py --runtime third_party/rlinf
```

The helper is idempotent. It refuses an incorrect HEAD, checksum mismatch,
partially applied patch or user edits to affected files. Unrelated working-tree
changes are preserved. It never resets, checks out or fetches a repository.

All behavioral changes are opt-in:

| Config | Behavior |
| --- | --- |
| `rollout.model.pass_predict_mode: true` | Pass train/eval to a dynamically registered adapter while preserving sampler kwargs |
| `actor.model.logprob_action_dim: 1` | Probability layout `[B,C,1]`, not physical control dimension 7; chunk-level loss sums the C step densities exactly once |
| `env.{train,eval}.xvla_absolute_control: true` | Reset/settling uses delta zero commands, then the child controller switches to `use_delta=False` before policy actions |
| `env.{train,eval}.xvla_raw_observations: true` | Pass raw main/wrist RGB; the adapter applies native main-only rotation |
| `env.{train,eval}.xvla_measured_state: true` | Return 20-D proprio from actual controller xyz and first two rotation-matrix columns, with zero gripper/padding |
| `env.{train,eval}.xvla_history_reset: true` | Emit a per-environment one-shot reset mask |
| `env.{train,eval}.xvla_strict_episodes: true` | Stop actual per-env execution after done/horizon, preserve chunk transport shape, and emit `executed_mask` for unexecuted suffixes |

Under `RLINF_XVLA_ISOLATED_RAY=1`, the driver starts a fresh `address="local"`
Ray instance for one Slurm-allocated node, without trying another user's Ray.
CPU count is bounded by `SLURM_CPUS_PER_TASK`, GPU count comes from the unchanged
`CUDA_VISIBLE_DEVICES`, and the object store is capped at 2 GiB. Set absolute,
job-ID-qualified `RAY_TMPDIR` and `FYP_XVLA_RAY_SPILL` paths. Worker construction
without an existing Ray context fails instead of falling back to `address=auto`.
Ray's code-sync environment and subsequent NodeProbe/manager initialization are
preserved. No global `ray stop` command is used.

The strict-episode path records cumulative actual policy and settling steps.
Masked suffixes do not execute a simulator action and have zero reward; this
prevents a synthetic negative relative reward from erasing a terminal success.
Existing model/controller/image defaults remain on the unmodified behavior path.

The patch does not implement X-VLA's probability model or losses. The adapter
must supply all-10D per-step density, preserve all auxiliary noise needed by its
declared strategy, and apply the execution mask consistently at recomputation.
An old/new likelihood ratio must never be built by truncating density to the
first seven physical control coordinates.

Local controller/reset tests use test doubles and do not prove MuJoCo dynamics.
Actual simulation, Ray transport and CUDA training require separate run evidence.
