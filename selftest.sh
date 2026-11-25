#!/bin/sh
# Software-only self-test (no board, no Vivado): synthetic random weights -> export -> C reference -> simulated board
# server -> benchmark -> comparison table. Takes about a minute. Needs python3 + numpy + gcc (Linux, macOS or WSL).
#   sh scripts/selftest.sh
set -e
cd "$(dirname "$0")/.."
T="${TMPDIR:-/tmp}/zcu104_selftest"; rm -rf "$T"; mkdir -p "$T"
python3 scripts/make_synthetic_export.py --out "$T/export"
python3 scripts/prepare_build.py --export "$T/export" --out "$T/build" > "$T/prep.log" && echo "prepare_build: OK"
sh board/build_cnnref.sh "$T/export"
python3 board/zcu_server.py --sim --qparams "$T/export/qparams.pkl" --port 5599 > "$T/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 3
python3 bench/bench_fpga.py --target 127.0.0.1:5599 --model_json "$T/export/fpga_model.json" --golden "$T/export/golden.npz" \
    --testset "$T/export/testset.npz" --n_real 20 --n_mixed 20 --bench_n 50 --out "$T/fpga_bench.json" | tee "$T/bench.log"
grep -q "FPGA matches numpy reference on 20/20 wafers; Cortex-A53 C software on 20/20" "$T/bench.log" && echo "SELFTEST PASSED" || { echo "SELFTEST FAILED"; exit 1; }
echo "(timing numbers and the 'implied clock' warning are expected here: the simulated board has no real timing)"
