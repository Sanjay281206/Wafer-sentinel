"""Create a fake-but-valid fpga_export/ (random weights) so you can test the whole ZCU104 tool chain BEFORE the real
model is exported, and so the software can be self-tested without any hardware.

  python scripts/make_synthetic_export.py --out fpga_export_synth

Writes the same files as make_int8_reference.py: weights.hex bias.hex mult.hex shift.hex weights.h golden/ golden.npz
fpga_model.json qparams.pkl, plus testset.npz (random maps; accuracy numbers from it are meaningless).
The logits of this random network are real integer-model outputs, so bit-exactness checks (RTL, FPGA, C) are fully valid.
"""
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host"))
from intmodel import (LAYERS, calibrate, quantize, int_forward, export_rtl, export_map_hex, export_c_header)

PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]


def fake_wafer(rng, fail_rate):
    yy, xx = np.mgrid[:64, :64]
    inside = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 < 30 ** 2
    m = np.where(rng.random((64, 64)) < fail_rate, 2, 1)
    return np.where(inside, m, 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fpga_export_synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_golden", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    folded = []
    for ic, oc, _ in LAYERS:
        w = (rng.standard_normal((oc, ic, 3, 3)) * np.sqrt(2.0 / (ic * 9))).astype(np.float32)
        b = (rng.standard_normal(oc) * 0.1).astype(np.float32)
        folded.append((w, b))
    fc = {"w": (rng.standard_normal((8, 64)) * 0.2).astype(np.float32), "b": (rng.standard_normal(8) * 0.1).astype(np.float32)}
    cal = np.stack([fake_wafer(rng, r) for r in rng.uniform(0.02, 0.4, 40)])
    qp = quantize(folded, fc, calibrate(cal, folded, fc))

    os.makedirs(os.path.join(a.out, "golden"), exist_ok=True)
    info = export_rtl(qp, a.out)
    export_c_header(qp, os.path.join(a.out, "weights.h"))
    gmaps = np.stack([fake_wafer(rng, r) for r in rng.uniform(0.02, 0.5, 2 * a.n_golden)]).astype(np.uint8)
    glog = np.array([int_forward(m, qp) for m in gmaps], dtype=np.int32)
    for i, (m, lg) in enumerate(zip(gmaps, glog)):
        export_map_hex(m, os.path.join(a.out, "golden", f"map_{i}.hex"))
        open(os.path.join(a.out, "golden", f"logits_{i}.txt"), "w").write(" ".join(str(int(v)) for v in lg) + "\n")
    np.savez(os.path.join(a.out, "golden.npz"), maps=gmaps, logits=glog)
    scale = np.array(qp["logit_scale"])
    thr = {p: 0.5 for p in PATTERNS}
    json.dump({"logit_scale": scale.tolist(), "thresholds_int_balanced": thr, "thresholds_int_sensitive": thr,
               "note": "SYNTHETIC random model, for tool-chain tests only"}, open(os.path.join(a.out, "fpga_model.json"), "w"), indent=2)
    pickle.dump({"qp": qp}, open(os.path.join(a.out, "qparams.pkl"), "wb"))

    n = a.n_test
    Xr = np.stack([fake_wafer(rng, r) for r in rng.uniform(0.02, 0.5, n)]); Xm = np.stack([fake_wafer(rng, r) for r in rng.uniform(0.02, 0.5, n)])
    np.savez_compressed(os.path.join(a.out, "testset.npz"), Xr=Xr, Xm=Xm,
                        Yr=(rng.random((n, 8)) < 0.15).astype(np.uint8), Ym=(rng.random((n, 8)) < 0.3).astype(np.uint8))
    print(f"synthetic export written to {a.out}/  ({info['words']} weight words, {info['groups']} lane groups, {len(gmaps)} golden wafers)")


if __name__ == "__main__":
    main()
