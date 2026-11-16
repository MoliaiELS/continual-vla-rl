#!/usr/bin/env bash
set -euo pipefail
: "${FYP_CONDA_ROOT:?}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_ROBOTWIN_ENV:?}" "${FYP_ROBOTWIN_CODE:?}"
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
data="$FYP_ROBOTWIN_DATA"
prefix="$FYP_ROBOTWIN_ENV"
code="$FYP_ROBOTWIN_CODE"
mkdir -p "$data"/{cache/pip,evidence,logs} "$code"

checkout() {
    local url="$1" rev="$2" path="$3" ref="${4:-$2}"
    if [[ ! -d "$path/.git" ]]; then git init -q "$path"; fi
    git -C "$path" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$path" fetch --depth=1 "$url" "$ref"
    git -C "$path" checkout -q --detach "$rev"
    test "$(git -C "$path" rev-parse HEAD)" = "$rev"
    git -C "$path" diff --quiet HEAD
}

case "${1:?prepare mode}" in
    assets)
        archives="$data/archives/$ASSETS_REV"
        tree="$data/sim-assets/$ASSETS_REV"
        "$FYP_CONDA_ROOT/envs/vla_budget_rl/bin/python" -u "$tool_dir/download_assets.py" "$archives" "$ASSETS_REV"
        if [[ ! -f "$tree/.extracted" ]]; then
            mkdir -p "$tree"
            for asset in embodiments objects background_texture; do
                unzip -q -o "$archives/$asset.zip" -d "$tree"
                test -d "$tree/$asset"
            done
            touch "$tree/.extracted"
        fi
        echo ROBOTWIN_ASSETS_READY
        ;;
    env)
        checkout https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_REV" "$code/RoboTwin-$ROBOTWIN_REV"
        checkout https://github.com/2toinf/X-VLA.git "$XVLA_REV" "$code/X-VLA-$XVLA_REV"
        checkout https://github.com/NVlabs/curobo.git "$CUROBO_REV" "$code/curobo-$CUROBO_REV" refs/tags/v0.7.8
        export CONDA_PKGS_DIRS="$FYP_CONDA_ROOT/pkgs-xvla:$FYP_CONDA_ROOT/pkgs-vla-budget:$FYP_CONDA_ROOT/pkgs"
        export PIP_CACHE_DIR="$data/cache/pip"
        if [[ ! -x "$prefix/bin/python" ]]; then
            "$FYP_CONDA_ROOT/bin/conda" create -y --prefix "$prefix" --clone "$FYP_CONDA_ROOT/envs/xvla_rlinf"
        fi
        "$FYP_CONDA_ROOT/bin/conda" install -y --prefix "$prefix" --override-channels \
            -c nvidia/label/cuda-12.4.1 -c conda-forge \
            cuda-nvcc=12.4.131 cuda-cudart-dev=12.4.127 cuda-cccl=12.4.127
        "$prefix/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
            -r "$tool_dir/requirements.txt" 'setuptools-scm==8.3.1'
        "$prefix/bin/python" "$tool_dir/patch_runtime.py"
        echo ROBOTWIN_ENV_READY
        ;;
    link)
        robot="$code/RoboTwin-$ROBOTWIN_REV"
        tree="$data/sim-assets/$ASSETS_REV"
        test -f "$tree/.extracted"
        for asset in embodiments objects background_texture; do
            test -d "$tree/$asset"
            if [[ ! -e "$robot/assets/$asset" ]]; then ln -s "$tree/$asset" "$robot/assets/$asset"; fi
            test "$(realpath "$robot/assets/$asset")" = "$tree/$asset"
        done
        (cd "$robot" && "$prefix/bin/python" script/update_embodiment_config_path.py)
        for arm in left right; do
            test -s "$robot/assets/embodiments/aloha-agilex/curobo_${arm}.yml"
        done
        echo ROBOTWIN_LINK_READY
        ;;
    *) exit 2 ;;
esac
