#!/usr/bin/env bash
# Start / stop a vLLM server in the background, for ldr's [collect] / [report] start_cmd and stop_cmd.
#   vllm-bg.sh start <name> <serve script> [args...]   runs the script detached, log in $LDR_VLLM_RUN/<name>.log
#   vllm-bg.sh stop  <name>                            stops it (TERM, then KILL after 60 s) and waits for the GPU
# The pid file lives in ${LDR_VLLM_RUN:-~/.cache/ldr-vllm}. systemd users can call `systemctl --user start ...` in
# start_cmd / stop_cmd instead.
set -euo pipefail
RUN=${LDR_VLLM_RUN:-$HOME/.cache/ldr-vllm}
mkdir -p "$RUN"
cmd=${1:?start|stop}; name=${2:?name}; pidf="$RUN/$name.pid"
case $cmd in
  start)
    shift 2
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then exit 0; fi
    # own process group (setsid, where it exists) so stop also ends vLLM's worker processes
    if command -v setsid >/dev/null; then nohup setsid "$@" >"$RUN/$name.log" 2>&1 < /dev/null &
    else nohup "$@" >"$RUN/$name.log" 2>&1 < /dev/null & fi
    echo $! >"$pidf" ;;
  stop)
    [ -f "$pidf" ] || exit 0
    pid=$(cat "$pidf")
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$pidf"
    sleep 5 ;;  # let the driver release the GPU memory before the next server loads
  *) echo "usage: $0 start|stop <name> [serve script args...]" >&2; exit 2 ;;
esac
