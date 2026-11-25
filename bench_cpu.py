"""Benchmark the WaferSentinel ONNX models on any CPU: Raspberry Pi 4, the ZCU104's Cortex-A53 cores, a laptop.
Needs only numpy and onnxruntime (on the ZCU104: pip3 install onnxruntime numpy).

  python bench_cpu.py --testset testset.npz --fp32 combined.onnx --int8 combined_int8.onnx --export_json combined_export.json --out zcu104_cpu_bench.json
  python bench_cpu.py ... --threads 1        # run again with 1 thread and with 4 threads, report both

For each model it reports accuracy on the SAME held-out wafers used on the laptop, latency (median and 95th
percentile, batch 1), throughput, model size and CPU temperature. Copy the table into section 7 of the report.
"""
import argparse, json, os, platform, subprocess, time
import numpy as np
import onnxruntime as ort

PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]


def to_input(maps):
    return np.stack([(maps == k) for k in (0, 1, 2)], axis=1).astype(np.float32)


def predict(sess, X, bs=64):
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(X), bs):
        z = sess.run(None, {name: to_input(X[i:i + bs])})[0]
        out.append(1 / (1 + np.exp(-z)))
    return np.concatenate(out)


def f1_per_class(Y, pred):
    tp = (pred * Y).sum(0); fp = (pred * (1 - Y)).sum(0); fn = ((1 - pred) * Y).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0), Y.sum(0)


def score(Pr, Yr, Pm, Ym, thr):
    pr = (Pr > thr).astype(int); pm = (Pm > thr).astype(int)
    fr, _ = f1_per_class(Yr, pr)
    fm, sm = f1_per_class(Ym, pm)
    clean = Yr.sum(1) == 0; called_clean = pr.sum(1) == 0
    k = Ym.sum(1)
    out = {"real_macroF1": fr.mean(), "real_false_alarm": (~called_clean[clean]).mean(),
           "real_miss": called_clean[~clean].mean(), "real_exact": (pr == Yr).all(1).mean(),
           "mixed_macroF1": fm[sm > 0].mean(), "mixed_exact": (pm == Ym).all(1).mean()}
    for g in (3, 4):
        m = k == g
        out[f"mixed_exact_{g}def"] = (pm[m] == Ym[m]).all(1).mean() if m.any() else float("nan")
    return out


def cpu_temp():
    import glob
    for p in ["/sys/class/thermal/thermal_zone0/temp"] + sorted(glob.glob("/sys/class/hwmon/hwmon*/temp1_input")):
        try:
            return int(open(p).read()) / 1000
        except Exception:
            pass
    return None


def throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "n/a (not a Raspberry Pi)"


def latency(sess, x1, n=300):
    name = sess.get_inputs()[0].name
    for _ in range(20):
        sess.run(None, {name: x1})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); sess.run(None, {name: x1}); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def batch_throughput(sess, X, bs=64, reps=5):
    name = sess.get_inputs()[0].name
    x = to_input(X[:bs]); sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {name: x})
    return bs * reps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="testset.npz")
    ap.add_argument("--fp32", default="combined.onnx")
    ap.add_argument("--int8", default="combined_int8.onnx")
    ap.add_argument("--export_json", default="combined_export.json")
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--out", default="cpu_bench.json")
    a = ap.parse_args()

    try:
        model = open("/proc/device-tree/model").read().strip("\x00 \n")
    except Exception:
        model = platform.platform()
    print(f"Device: {model} | machine: {platform.machine()} | onnxruntime {ort.__version__} | threads: {a.threads or 'default'}")
    print(f"CPU temperature at start: {cpu_temp()} C | throttle flags: {throttled()}\n")

    d = np.load(a.testset)
    Xr, Yr, Xm, Ym = d["Xr"], d["Yr"].astype(int), d["Xm"], d["Ym"].astype(int)
    js = json.load(open(a.export_json))
    thr = {k: np.array([js[k][p] for p in PATTERNS]) for k in ("thresholds_fp32", "thresholds_int8", "thresholds_int8_sensitive")}

    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    sessions = {"fp32": ort.InferenceSession(a.fp32, so, providers=["CPUExecutionProvider"]),
                "int8": ort.InferenceSession(a.int8, so, providers=["CPUExecutionProvider"])}
    rows = {}
    for label, which, key in [("FP32 (balanced)", "fp32", "thresholds_fp32"),
                              ("INT8 (balanced)", "int8", "thresholds_int8"),
                              ("INT8 (sensitive)", "int8", "thresholds_int8_sensitive")]:
        s = sessions[which]
        r = score(predict(s, Xr), Yr, predict(s, Xm), Ym, thr[key])
        if "sensitive" not in label:                      # latency depends on the model only
            r["latency_median_ms"], r["latency_p95_ms"] = latency(s, to_input(Xr[:1]))
            r["throughput_wafers_s"] = batch_throughput(s, Xr)
            r["size_KB"] = os.path.getsize(a.fp32 if which == "fp32" else a.int8) / 1024
        rows[label] = {k: float(v) for k, v in r.items()}

    cols = ["real_macroF1", "real_false_alarm", "real_miss", "real_exact", "mixed_exact", "mixed_exact_3def", "mixed_exact_4def"]
    print(f"{'model':<18}" + "".join(f"{c:>18}" for c in cols))
    for label, r in rows.items():
        print(f"{label:<18}" + "".join(f"{r[c]:>18.3f}" for c in cols))
    print()
    for label, r in rows.items():
        if "latency_median_ms" in r:
            print(f"{label:<18} latency median {r['latency_median_ms']:.2f} ms | p95 {r['latency_p95_ms']:.2f} ms | "
                  f"batch-64 throughput {r['throughput_wafers_s']:.0f} wafers/s | size {r['size_KB']:.0f} KB")
    print(f"\nCPU temperature at end: {cpu_temp()} C | throttle flags: {throttled()}  (0x0 means no throttling)")
    json.dump({"device": model, "threads": a.threads, "rows": rows}, open(a.out, "w"), indent=2)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
