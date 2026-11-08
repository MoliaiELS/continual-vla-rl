# Audited X-VLA model namespace

Source: `2toinf/X-VLA@6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`.
The eight Python files come from its `models/` directory. The original Apache-2.0
headers and [LICENSE](LICENSE) are retained. Exact upstream and vendored content
hashes are recorded in [UPSTREAM.json](UPSTREAM.json).

The only local vendor patch defers FastAPI, uvicorn, json_numpy, OpenCV and PIL
imports to the optional HTTP service methods. The model's forward, native clean
action prediction loop, preprocessing and supervised losses are unchanged.
Training imports this explicit namespace; it does not execute checkpoint custom
Python or use a top-level `models` / `datasets` package.

`XVLA.generate_actions()` remains the immutable native reference. Differentiable
RL sampling belongs to the adapter, with separately documented probability
semantics. `EE6DActionSpace` retains clean-action supervision: position MSE ×500,
Rotate6D MSE ×10, and mean binary cross entropy for both gripper channels.

The upstream `LiberoHandler` expects already-converted absolute controller
targets (`abs_action_6d`), drops the first image, resamples a one-second window
into `num_actions+1` points, and takes the first point as proprio. Raw relative
LIBERO actions must be replayed to obtain controller goal poses before this
boundary; the adapter must never relabel them as absolute poses.
