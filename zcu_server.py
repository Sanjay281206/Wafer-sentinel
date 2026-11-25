"""WaferSentinel server for the ZCU104. Runs on the board (as root) and speaks the SAME binary protocol as the
ZedBoard firmware, but over TCP instead of the serial line, so the host software changes only in its transport.

  sudo python3 zcu_server.py --bitstream ~/zcu104_wafersentinel/output/wafer_zcu104.bit --cnnref libcnnref.so
  python3 zcu_server.py --sim --qparams ../fpga_export/qparams.pkl          # no hardware: register-level emulation

Request : 0xA5, cmd, 4096 map bytes (values 0..2) [, uint32 n for cmd 3]                         (little-endian)
  cmd 1 = run on the FPGA accelerator, cmd 2 = C reference on the Cortex-A53 (libcnnref.so),
  cmd 3 = run the accelerator n times back to back (sustained speed / power measurement)
Response: 0xA5, status (0 ok, 1 timeout), 8 x int32 logits, uint32 accelerator cycles, uint64 time ticks,
          uint32 ticks per second                                                                (50 bytes)
Here a "tick" is one nanosecond (ticks per second = 1e9). The timed part of cmd 1/3 is start -> done only; the
image upload is not included (same as the ZedBoard firmware).
"""
import argparse
import ctypes
import os
import pickle
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zcu_accel import (DEFAULT_BASE, MockMmio, ZcuAccel, fpga_state, load_bitstream, pl_clock_mhz)

RESP = struct.Struct("<BB8iIQI")


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def make_cnnref(path, qp=None):
    """Returns f(map) -> int32[8]: the plain-C reference compiled for the A53 (preferred), else the numpy model."""
    if path and os.path.exists(path):
        lib = ctypes.CDLL(os.path.abspath(path))
        lib.cnn_infer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cnn_infer.restype = None

        def f(m):
            a = np.ascontiguousarray(m, np.uint8)
            out = np.zeros(8, np.int32)
            lib.cnn_infer(a.ctypes.data, out.ctypes.data)
            return out
        return f, "C (" + os.path.basename(path) + ")"
    if qp is not None:
        from intmodel import int_forward
        return (lambda m: np.asarray(int_forward(np.asarray(m, np.uint8).reshape(64, 64), qp), np.int32)), \
            "numpy fallback (NOT comparable to the C baseline)"
    return None, "none"


def serve(conn, acc, sw):
    while True:
        while recv_exact(conn, 1) != b"\xa5":
            pass
        cmd = recv_exact(conn, 1)[0]
        m = np.frombuffer(recv_exact(conn, 4096), np.uint8)
        n = struct.unpack("<I", recv_exact(conn, 4))[0] if cmd == 3 else 1
        status, cycles, ticks, logits = 0, 0, 0, np.zeros(8, np.int32)
        try:
            if cmd in (1, 3):
                acc.load_image(m)
                t0 = time.perf_counter_ns()
                for _ in range(n):
                    acc.run()
                ticks = time.perf_counter_ns() - t0
                cycles, logits = acc.cycles(), acc.logits()
            else:
                if sw is None:
                    raise RuntimeError("no software baseline available (build libcnnref.so, see board/build_cnnref.sh)")
                t0 = time.perf_counter_ns()
                logits = sw(m.reshape(64, 64))
                ticks = time.perf_counter_ns() - t0
        except TimeoutError:
            status = 1
        conn.sendall(RESP.pack(0xA5, status, *[int(v) for v in logits], cycles, ticks, 1_000_000_000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=DEFAULT_BASE, help="accelerator base address (Vivado Address Editor)")
    ap.add_argument("--bitstream", default=None, help="PYNQ: wafer_zcu104.bit (+ .hwh next to it); other Linux: wafer_zcu104.bit.bin")
    ap.add_argument("--cnnref", default="libcnnref.so", help="C reference shared library for the Cortex-A53 baseline")
    ap.add_argument("--sim", action="store_true", help="no hardware: emulate the register map (timings are NOT real)")
    ap.add_argument("--qparams", default="../fpga_export/qparams.pkl", help="needed for --sim")
    ap.add_argument("--force", action="store_true", help="skip the 'PL is programmed' safety check")
    a = ap.parse_args()

    qp = None
    if a.qparams and os.path.exists(a.qparams):
        qp = pickle.load(open(a.qparams, "rb"))["qp"]
    if a.sim:
        if qp is None:
            sys.exit("--sim needs --qparams fpga_export/qparams.pkl")
        acc = ZcuAccel(MockMmio(qp))
        print("SIMULATION MODE: no hardware is used, timing numbers are meaningless")
    else:
        overlay = None
        if a.bitstream:
            print("loading", a.bitstream)
            overlay = load_bitstream(a.bitstream)          # keep the reference alive for the life of the server
        print("fpga_manager state:", fpga_state(), "| PL clock 0:", pl_clock_mhz() or "unknown", "MHz")
        acc = ZcuAccel.open(a.base, a.force)
        print(f"accelerator found at 0x{a.base:08x}")
    sw, sw_name = make_cnnref(a.cnnref, qp if a.sim else None)
    print("software baseline:", sw_name)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(1)
    print(f"listening on {a.host}:{a.port}  (one client at a time; Ctrl-C to stop)", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("client", addr, flush=True)
        try:
            serve(conn, acc, sw)
        except (EOFError, ConnectionError):
            print("client left", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
