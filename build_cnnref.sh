#!/bin/sh
# Build the plain-C software baseline for the Cortex-A53 as a shared library (run on the ZCU104, or anywhere with gcc).
#   sh build_cnnref.sh ../fpga_export          # needs weights.h from make_int8_reference.py in that folder
# Same C source as the bare-metal firmware (firmware_baremetal/cnn_ref.c); -O2 is the compiler setting used for the baseline.
set -e
EXPORT="${1:-../fpga_export}"
[ -f "$EXPORT/weights.h" ] || { echo "weights.h not found in $EXPORT"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
gcc -O2 -shared -fPIC -I"$EXPORT" -o "$HERE/libcnnref.so" "$HERE/../firmware_baremetal/cnn_ref.c"
echo "built $HERE/libcnnref.so  (gcc -O2)"
