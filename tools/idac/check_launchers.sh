#!/usr/bin/env bash
set -euo pipefail
# A deliberate Python failure must survive tee, including argument boundaries.
repo="$(cd "$(dirname "$0")/../.." && pwd)"
testdir="$(mktemp -d "${TMPDIR:-/tmp}/fyp-launchers.XXXXXX")"
mkdir -p "$testdir/bin"
cat > "$testdir/bin/python" <<'PY'
#!/usr/bin/env bash
printf 'ARG=<%s>\n' "$@"
echo "LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
exit 23
PY
chmod +x "$testdir/bin/python"
export PATH="$testdir/bin:$PATH" LIBERO_CONFIG_PATH="$testdir/libero config"
for launcher in run_embodiment eval_embodiment; do
    export LOG_DIR="$testdir/$launcher logs"
    if bash "$repo/examples/embodiment/$launcher.sh" fyp/spatial_smoke 'runner.logger.project_name=with spaces' > "$testdir/$launcher.txt" 2>&1; then
        echo "Failure swallowed by $launcher" >&2; exit 1
    else
        status=$?
        test "$status" = 23
    fi
    grep -Fx 'ARG=<runner.logger.project_name=with spaces>' "$testdir/$launcher.txt"
    grep -Fx "LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH" "$testdir/$launcher.txt"
done
echo "LAUNCHER_FAILURE_PROPAGATION_OK evidence=$testdir"
