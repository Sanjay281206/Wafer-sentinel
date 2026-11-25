"""Put the benchmark results of all platforms into one comparison table (report section 7).

  python compare_results.py --pi pi_bench.json --zcu_cpu zcu104_cpu_bench.json --fpga fpga_bench.json \
         [--zed zed_fpga_bench.json] --power pi4=5.9 zcu104_cpu=4.1 zcu104_fpga=2.3 zed=1.4 --out comparison.md

Inputs: bench_cpu.py outputs (Raspberry Pi 4 and/or ZCU104 Cortex-A53), bench_fpga.py outputs (ZCU104, and the old ZedBoard
run if you still have it). --power gives watts per platform (added power under load, from power_log.py or a meter);
energy per wafer = power x time per wafer. All inputs are optional; missing platforms are left out.
"""
import argparse, json


def cpu_rows(path, name, power):
    d = json.load(open(path)); rows = []
    for label, r in d["rows"].items():
        if "latency_median_ms" not in r:
            continue
        rows.append({"platform": f"{name} ONNX {label.split()[0]}", "lat_ms": r["latency_median_ms"],
                     "thr": r["throughput_wafers_s"], "f1": r["real_macroF1"], "key": name, "note": f"p95 {r['latency_p95_ms']:.1f} ms; batch-64 throughput; {d['device']}"})
    return rows


def fpga_rows(path, name, key, sw_key):
    d = json.load(open(path)); t = d["timing"]; acc = d.get("accuracy_balanced", {})
    f1 = acc.get("real_macroF1", float("nan"))
    rows = [{"platform": f"{name} accelerator (PL, {t['accel_clock_mhz']:.0f} MHz)", "lat_ms": t["accel_compute_ms"],
             "thr": t["sustained_wafers_per_s"], "f1": f1, "key": key, "note": f"{t['accel_cycles']} cycles; sustained over many runs; accuracy on a subset"}]
    if t["arm_c_software_ms"] == t["arm_c_software_ms"]:
        rows.append({"platform": f"{name} Arm core, C software", "lat_ms": t["arm_c_software_ms"], "thr": 1e3 / t["arm_c_software_ms"],
                     "f1": f1, "key": sw_key, "note": "same integer network in plain C"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi"); ap.add_argument("--zcu_cpu"); ap.add_argument("--fpga"); ap.add_argument("--zed")
    ap.add_argument("--power", nargs="*", default=[], help="platform=watts, platforms: pi4 zcu104_cpu zcu104_fpga zed")
    ap.add_argument("--out", default="comparison.md")
    a = ap.parse_args()
    pw = {k: float(v) for k, v in (p.split("=") for p in a.power)}
    rows = []
    if a.pi: rows += cpu_rows(a.pi, "pi4", None)
    if a.zcu_cpu: rows += cpu_rows(a.zcu_cpu, "zcu104_cpu", None)
    if a.zed: rows += fpga_rows(a.zed, "ZedBoard", "zed", "zed")
    if a.fpga: rows += fpga_rows(a.fpga, "ZCU104", "zcu104_fpga", "zcu104_cpu")
    pretty = {"pi4": "Raspberry Pi 4", "zcu104_cpu": "ZCU104 Cortex-A53"}
    lines = ["| Platform | Latency per wafer (ms) | Throughput (wafers/s) | Real macro-F1 | Power (W) | Energy per wafer (mJ) | Notes |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        p = pw.get(r["key"])
        e = f"{p * r['lat_ms']:.1f}" if p is not None else "-"
        name = r["platform"]
        for k, v in pretty.items():
            name = name.replace(k, v)
        lines.append(f"| {name} | {r['lat_ms']:.2f} | {r['thr']:.0f} | {r['f1']:.3f} | {p if p is not None else '-'} | {e} | {r['note']} |")
    md = "\n".join(lines)
    print(md)
    open(a.out, "w").write(md + "\n")
    print("\nsaved", a.out)
    print("Latency for CPUs = batch-1 median; for the accelerator = compute time (no link). Compare end-to-end times separately (bench_fpga.py).")


if __name__ == "__main__":
    main()
