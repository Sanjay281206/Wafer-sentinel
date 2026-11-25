"""Driver for the WaferSentinel CNN accelerator on the ZCU104 (runs on the board's Linux, needs root).

The accelerator sits behind the PS master port M_AXI_HPM0_FPD. Vivado normally places it at 0xA0000000 (check the
Address Editor / the build log). Register map (byte offsets, 8 KB window):
  0x0000 CTRL   write 1 -> start        0x0004 STATUS  bit0 busy, bit1 done      0x0008 CYCLES  cycles of the last run
  0x0010 + 4*i  LOGIT i (int32)         0x1000 + 4*j   image word j (4 die values per word, j = 0..1023)

IMPORTANT: reading or writing this address range while the PL is NOT programmed (or has no clock) hangs the AXI bus
and can freeze the board. Always load the bitstream first; ZcuAccel.open() refuses to touch the bus unless
/sys/class/fpga_manager/fpga0/state says "operating" (or the state file does not exist).
"""
import mmap
import os
import subprocess
import time

import numpy as np

REG_CTRL, REG_STATUS, REG_CYCLES, REG_LOGIT, REG_IMG = 0x0000, 0x0004, 0x0008, 0x0010, 0x1000
DEFAULT_BASE = 0xA0000000
WINDOW = 0x2000


# ------------------------------------------------------------------ register access back-ends
class DevMemMmio:
    """32-bit register access through /dev/mem (works on PYNQ and on PetaLinux; needs root)."""

    def __init__(self, base=DEFAULT_BASE, size=WINDOW):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.w = np.frombuffer(self.mm, dtype=np.uint32)

    def read(self, off):
        return int(self.w[off >> 2])

    def write(self, off, value):
        self.w[off >> 2] = value & 0xFFFFFFFF


class MockMmio:
    """Register-level model of wafer_cnn_axil. No hardware needed: used by `zcu_server.py --sim` to test the driver,
    the server and the host software anywhere. Logits come from the numpy integer model (bit-exact with the FPGA)."""

    def __init__(self, qp, cycles=513864, busy_reads=3):
        from intmodel import int_forward
        self.qp, self.fwd, self.cycles, self.busy_reads = qp, int_forward, cycles, busy_reads
        self.img = np.zeros(1024, np.uint32)
        self.logit = np.zeros(8, np.int32)
        self.left = 0
        self.done = 0

    def write(self, off, value):
        value &= 0xFFFFFFFF
        if off >= REG_IMG:
            self.img[(off - REG_IMG) >> 2] = value
        elif off == REG_CTRL and (value & 1):
            m = self.img.astype("<u4").view(np.uint8).reshape(64, 64)
            self.logit = np.asarray(self.fwd(m, self.qp), np.int32)
            self.left, self.done = self.busy_reads, 0

    def read(self, off):
        if off == REG_STATUS:
            if self.left > 0:
                self.left -= 1
                if self.left == 0:
                    self.done = 1
                return 1
            return 2 if self.done else 0
        if off == REG_CYCLES:
            return self.cycles
        if REG_LOGIT <= off < REG_LOGIT + 32:
            return int(self.logit[(off - REG_LOGIT) >> 2]) & 0xFFFFFFFF
        return 0


# ------------------------------------------------------------------ PL (bitstream) helpers
def fpga_state():
    try:
        return open("/sys/class/fpga_manager/fpga0/state").read().strip()
    except OSError:
        return "unknown"


def load_bitstream(path):
    """PYNQ: path to <name>.bit (the <name>.hwh must sit next to it) -> pynq.Overlay, which also sets the PL clock.
    Other Linux: path to <name>.bit.bin -> fpgautil. Returns the Overlay object (or None)."""
    if path.endswith(".bin"):
        subprocess.run(["fpgautil", "-b", path], check=True)
        return None
    from pynq import Overlay
    return Overlay(path)


def pl_clock_mhz():
    """PL clock 0 in MHz when pynq is available (None otherwise)."""
    try:
        from pynq import Clocks
        return float(Clocks.fclk0_mhz)
    except Exception:
        return None


# ------------------------------------------------------------------ accelerator
class ZcuAccel:
    def __init__(self, mmio):
        self.m = mmio

    @classmethod
    def open(cls, base=DEFAULT_BASE, force=False):
        st = fpga_state()
        if st not in ("operating", "unknown") and not force:
            raise RuntimeError(f"PL is not programmed (fpga_manager state: {st!r}). Load the bitstream first "
                               "(zcu_server.py --bitstream ...), or use --force if you are sure.")
        acc = cls(DevMemMmio(base))
        acc.probe()
        return acc

    def probe(self):
        st = self.m.read(REG_STATUS)
        if st > 3:
            raise RuntimeError(f"unexpected STATUS 0x{st:08x}: no accelerator at this address (check the base address)")
        return st

    def load_image(self, m):
        a = np.ascontiguousarray(m, dtype=np.uint8).reshape(-1)
        if a.size != 4096:
            raise ValueError("the wafer map must be 64 x 64")
        for i, w in enumerate(a.view("<u4").tolist()):
            self.m.write(REG_IMG + 4 * i, w)

    def run(self, timeout_s=5.0):
        """Start and wait for the end (waits for busy to rise, then fall, so a stale 'done' is never used)."""
        m, t_end = self.m, time.perf_counter() + timeout_s
        m.write(REG_CTRL, 1)
        while not (m.read(REG_STATUS) & 1):
            if time.perf_counter() > t_end:
                raise TimeoutError("accelerator did not start")
        while m.read(REG_STATUS) & 1:
            if time.perf_counter() > t_end:
                raise TimeoutError("accelerator did not finish")

    def cycles(self):
        return self.m.read(REG_CYCLES)

    def logits(self):
        return np.array([self.m.read(REG_LOGIT + 4 * i) for i in range(8)], dtype=np.uint32).view(np.int32)
