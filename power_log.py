"""Log the ZCU104's on-board power monitors (INA226 chips, exposed by Linux as hwmon devices). Run on the board.

  sudo python3 power_log.py --list                               # what sensors does this image expose?
  sudo python3 power_log.py --seconds 20 --label idle --out idle.json
  sudo python3 power_log.py --seconds 25 --label fpga --out fpga.json     # while:  bench_fpga.py ... --bench_n 5000
  python3 power_log.py --diff idle.json fpga.json                # per-rail difference and total

Each channel is one rail of the board (the names/labels depend on the image; match them with the ZCU104 schematic or
UG1267, e.g. the PL core rail VCCINT). 'Added power' = (rail power under load) - (rail power idle), summed over all
rails, is the number to quote for the accelerator or for the A53 workload. Idle = bitstream loaded, server running,
nothing being computed. Sensors without a power1_input file are skipped. If --list shows nothing, use a USB/inline
power meter on the 12 V input instead (same idle/load method) or `pynq.get_rails()` on a PYNQ image.
"""
import argparse, glob, json, os, statistics, time


def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None


def sensors():
    out = []
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = rd(d + "/name") or "?"
        label = rd(d + "/label") or rd(d + "/power1_label")
        has_p = os.path.exists(d + "/power1_input")
        out.append({"dir": d, "name": name, "label": label or f"{name}@{os.path.basename(d)}", "power": has_p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", nargs=2, metavar=("IDLE_JSON", "LOAD_JSON"))
    a = ap.parse_args()

    if a.diff:
        i, l = (json.load(open(f)) for f in a.diff)
        tot = 0.0
        print(f"{'rail':<34}{'idle W':>9}{'load W':>9}{'added W':>10}")
        for k in l["mean_w"]:
            d = l["mean_w"][k] - i["mean_w"].get(k, 0.0)
            tot += d
            print(f"{k:<34}{i['mean_w'].get(k, 0):>9.3f}{l['mean_w'][k]:>9.3f}{d:>10.3f}")
        print(f"{'TOTAL (all rails)':<34}{sum(i['mean_w'].values()):>9.3f}{sum(l['mean_w'].values()):>9.3f}{tot:>10.3f}")
        return

    ss = sensors()
    if a.list or not any(s["power"] for s in ss):
        for s in ss:
            print(f"{s['dir']:<28}{s['name']:<14}{s['label']:<30}{'power1_input' if s['power'] else '-'}")
        if not any(s["power"] for s in ss):
            print("\nNo hwmon power sensors found: use an external power meter or pynq.get_rails().")
        if a.list or not any(s["power"] for s in ss):
            return

    use = [s for s in ss if s["power"]]
    series = {s["label"]: [] for s in use}
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        for s in use:
            v = rd(s["dir"] + "/power1_input")
            if v is not None:
                series[s["label"]].append(int(v) / 1e6)            # microwatt -> watt
        time.sleep(1.0 / a.hz)
    res = {"label": a.label, "seconds": a.seconds, "mean_w": {k: statistics.mean(v) for k, v in series.items() if v},
           "max_w": {k: max(v) for k, v in series.items() if v}}
    for k, v in res["mean_w"].items():
        print(f"{k:<34}{v:>8.3f} W mean   {res['max_w'][k]:>8.3f} W max")
    print(f"{'TOTAL (sum of rails)':<34}{sum(res['mean_w'].values()):>8.3f} W mean")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print("saved", a.out)


if __name__ == "__main__":
    main()
