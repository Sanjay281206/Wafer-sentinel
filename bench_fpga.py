"""Benchmark and validate the ZCU104 accelerator. Runs on the laptop (TCP to the board) or on the board itself
(connect to localhost). Needs numpy; pyserial only if you use the bare-metal serial firmware.

  python bench_fpga.py --target 192.168.2.99:5555 --model_json fpga_model.json --golden golden.npz --testset testset.npz --clock_mhz 100
  python bench_fpga.py --target /dev/ttyUSB1 ...        # bare-metal firmware over the USB-UART (115200 baud: slow)

1. Golden check: FPGA, Cortex-A53 C software and the numpy reference must give IDENTICAL logits.
2. Accuracy of the FPGA path on real and mixed test wafers.
3. Timing: accelerator compute time, A53 C software time, sustained throughput, end-to-end time including the link.
   The sustained run also gives an IMPLIED PL clock (cycles / time); if it disagrees with --clock_mhz the script warns.
Prints the table for section 7 of the report and saves it to fpga_bench.json (default).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from wafersentinel.fpga_backend import FpgaBackend
except ImportError:
    try:
        from fpga_backend import FpgaBackend
    except ImportError:                                   # run straight from the unzipped package
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host"))
        from fpga_backend import FpgaBackend

PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]


def f1_per_class(Y, pred):
    tp = (pred * Y).sum(0); fp = (pred * (1 - Y)).sum(0); fn = ((1 - pred) * Y).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0), Y.sum(0)


def metrics(Pr, Yr, Pm, Ym, thr):
    out = {}
    if len(Pr):
        pr = (Pr > thr).astype(int); fr, _ = f1_per_class(Yr, pr)
        clean = Yr.sum(1) == 0; cc = pr.sum(1) == 0
        out.update(real_macroF1=fr.mean(), real_false_alarm=(~cc[clean]).mean() if clean.any() else float("nan"),
                   real_miss=cc[~clean].mean() if (~clean).any() else float("nan"), real_exact=(pr == Yr).all(1).mean())
    if len(Pm):
        pm = (Pm > thr).astype(int); fm, sm = f1_per_class(Ym, pm)
        out.update(mixed_macroF1=fm[sm > 0].mean(), mixed_exact=(pm == Ym).all(1).mean())
    return {k: float(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", "--port", dest="target", required=True,
                    help="host:port of zcu_server.py (e.g. 192.168.2.99:5555) or a serial device (/dev/ttyUSB1, COM5)")
    ap.add_argument("--baud", type=int, default=921600, help="serial transport only; must equal UART_BAUD in main.c")
    ap.add_argument("--model_json", default="fpga_model.json"); ap.add_argument("--golden", default="golden.npz")
    ap.add_argument("--testset", default="testset.npz")
    ap.add_argument("--n_real", type=int, default=300); ap.add_argument("--n_mixed", type=int, default=300)
    ap.add_argument("--bench_n", type=int, default=200); ap.add_argument("--clock_mhz", type=float, default=100.0)
    ap.add_argument("--sw_runs", type=int, default=3); ap.add_argument("--out", default="fpga_bench.json")
    a = ap.parse_args()

    fpga = FpgaBackend(a.target, a.baud, a.model_json)
    meta = json.load(open(a.model_json))
    res = {}

    # ---- 1. golden check
    g = np.load(a.golden); ok_hw = ok_sw = 0
    for m, ref in zip(g["maps"], g["logits"]):
        hw = fpga.infer(m); sw = fpga.infer_sw(m) if a.sw_runs > 0 else None
        ok_hw += int(hw["status"] == 0 and np.array_equal(hw["logits"], ref))
        ok_sw += int(sw is not None and np.array_equal(sw["logits"], ref))
    n_g = len(g["maps"])
    print(f"GOLDEN CHECK: FPGA matches numpy reference on {ok_hw}/{n_g} wafers; Cortex-A53 C software on {ok_sw}/{n_g}")
    res["golden"] = {"n": n_g, "fpga_match": ok_hw, "arm_sw_match": ok_sw}

    # ---- 2. accuracy on test wafers
    d = np.load(a.testset)
    rng = np.random.default_rng(0)
    ir = rng.choice(len(d["Xr"]), min(a.n_real, len(d["Xr"])), replace=False)
    im = rng.choice(len(d["Xm"]), min(a.n_mixed, len(d["Xm"])), replace=False)
    rows = {"real": [], "mixed": []}; rt = []; cyc = []
    for key, X, idx in (("real", d["Xr"], ir), ("mixed", d["Xm"], im)):
        for i in idx:
            r = fpga.infer(X[i]); rows[key].append(fpga.probs(r["logits"])); rt.append(r["round_trip_s"]); cyc.append(r["accel_cycles"])
    Yr, Ym = d["Yr"][ir].astype(int), d["Ym"][im].astype(int)
    Pr, Pm = np.array(rows["real"]).reshape(-1, 8), np.array(rows["mixed"]).reshape(-1, 8)
    for mode in ("balanced", "sensitive"):
        thr = np.array([meta[f"thresholds_int_{mode}"][p] for p in PATTERNS])
        res[f"accuracy_{mode}"] = metrics(Pr, Yr, Pm, Ym, thr)
        print(f"ACCURACY ({mode}, {len(ir)} real + {len(im)} mixed wafers):", {k: round(v, 3) for k, v in res[f'accuracy_{mode}'].items()})

    # ---- 3. timing
    m0 = d["Xr"][0]
    cycles = int(np.median(cyc)); hw_ms = cycles / (a.clock_mhz * 1e3)
    sw = [fpga.infer_sw(m0) for _ in range(a.sw_runs)] if a.sw_runs > 0 else []
    sw_ms = float(np.median([s["arm_ticks"] / s["ticks_per_s"] * 1e3 for s in sw])) if sw else float("nan")
    b = fpga.bench(m0, a.bench_n); sust_ms = b["arm_ticks"] / b["ticks_per_s"] * 1e3 / a.bench_n
    implied = cycles / (sust_ms * 1e3)                   # MHz the accelerator really ran at (overhead makes it slightly low)
    res["timing"] = {"accel_cycles": cycles, "accel_clock_mhz": a.clock_mhz, "accel_compute_ms": hw_ms,
                     "arm_c_software_ms": sw_ms, "sustained_ms_per_wafer": sust_ms,
                     "sustained_wafers_per_s": 1e3 / sust_ms, "end_to_end_median_ms": float(np.median(rt)) * 1e3,
                     "transport": "serial" if fpga.serial else "tcp", "target": a.target,
                     "speedup_vs_arm_c": sw_ms / hw_ms if sw else float("nan"), "implied_clock_mhz": implied}
    t = res["timing"]
    if abs(implied - a.clock_mhz) / a.clock_mhz > 0.05:
        print(f"\nWARNING: cycles / sustained time implies a PL clock of about {implied:.1f} MHz, not {a.clock_mhz:.0f} MHz. "
              f"Re-run with --clock_mhz {implied:.0f} (or check the PL clock) before quoting accelerator times.")
    link = f"serial {a.baud} baud" if fpga.serial else "TCP"
    print("\n| Platform | Compute time per wafer | Notes |\n|---|---|---|")
    print(f"| ZCU104 accelerator (PL, {a.clock_mhz:.0f} MHz) | {t['accel_compute_ms']:.2f} ms ({cycles} cycles) | sustained {t['sustained_wafers_per_s']:.0f} wafers/s, measured over {a.bench_n} runs |")
    print(f"| ZCU104 Cortex-A53, C software | {t['arm_c_software_ms']:.0f} ms | plain C, not optimised for the core |")
    print(f"| End to end incl. link ({link}) | {t['end_to_end_median_ms']:.1f} ms | median over the accuracy runs |")
    json.dump(res, open(a.out, "w"), indent=2)
    print("\nSaved", a.out, "- for power: run  python bench_fpga.py ... --bench_n 5000  while board/power_log.py (or a meter) records")


if __name__ == "__main__":
    main()
