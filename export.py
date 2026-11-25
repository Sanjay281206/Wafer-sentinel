"""Phase 3: export to ONNX, quantise to INT8 with REAL calibration wafers, and compare
FP32 vs INT8 accuracy + latency on the clean held-out test half (same split as evaluate.py).

  python -m wafersentinel.export --ckpt runs/wm811k_e60.pt --pkl data/LSWMD_labelled.pkl

Outputs go to <ckpt folder>/export/ :
  <name>.onnx            FP32, dynamic batch (for evaluation)
  <name>_b1.onnx         FP32, fixed batch 1 (feed this to the Hailo parser)
  <name>_int8.onnx       INT8 QDQ model (for ONNX Runtime on the Pi CPU)
  <name>_calib_nhwc.npy  calibration set in Hailo's channels-last layout (N,H,W,3)
  <name>_export.json     all numbers for your report
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from .classes import PATTERNS
from .data import load_wm811k, to_tensor_input
from .model import WaferCNN
from .evaluate import tune


class Reader(CalibrationDataReader):
    def __init__(self, arr):
        self.it = iter(arr)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"wafer": x[None].astype(np.float32)}


def export_onnx(model, size, path, dynamic):
    dummy = torch.zeros(1, 3, size, size)
    kw = dict(input_names=["wafer"], output_names=["logits"], opset_version=13)
    if dynamic:
        kw["dynamic_axes"] = {"wafer": {0: "b"}, "logits": {0: "b"}}
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)   # newer torch: use the classic exporter
    except TypeError:
        torch.onnx.export(model, dummy, path, **kw)                 # older torch has no 'dynamo' argument


def predict_onnx(path, X, bs=256):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    out = []
    for i in range(0, len(X), bs):
        z = s.run(None, {name: to_tensor_input(X[i:i + bs])})[0]
        out.append(1 / (1 + np.exp(-z)))
    return np.concatenate(out)


def metrics(P, Y, thr):
    pred = (P > thr).astype(int)
    f = precision_recall_fscore_support(Y, pred, zero_division=0)[2]
    clean = Y.sum(1) == 0
    called_clean = pred.sum(1) == 0
    return {"macro_f1": float(f.mean()), "per_class_f1": f,
            "false_alarm": float((~called_clean[clean]).mean()),
            "miss": float(called_clean[~clean].mean()),
            "exact": float((pred == Y).all(1).mean())}


def latency_ms(path, x1, n=300):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    for _ in range(20):
        s.run(None, {name: x1})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(None, {name: x1}); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/wm811k_e60.pt")
    ap.add_argument("--pkl", default="data/LSWMD_labelled.pkl")
    ap.add_argument("--per_class", type=int, default=40, help="calibration wafers per class")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    size = ck["size"]
    model = WaferCNN(len(PATTERNS)); model.load_state_dict(ck["state"]); model.eval()
    name = os.path.splitext(os.path.basename(a.ckpt))[0]
    out_dir = os.path.join(os.path.dirname(a.ckpt) or ".", "export"); os.makedirs(out_dir, exist_ok=True)
    p = lambda suffix: os.path.join(out_dir, name + suffix)

    # --- same splits as train.py / evaluate.py ---
    X, Y, G = load_wm811k(a.pkl, size, return_groups=True)
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, Y, G))
    Xv, Yv, Gv = X[va], Y[va], G[va]
    tu, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=1).split(Xv, Yv, Gv))

    # --- ONNX FP32 ---
    export_onnx(model, size, p(".onnx"), dynamic=True)
    export_onnx(model, size, p("_b1.onnx"), dynamic=False)

    # --- calibration set: real TRAINING wafers, class-balanced (rare classes included) ---
    rng = np.random.default_rng(0)
    pick = []
    for c in range(len(PATTERNS)):
        idx = tr[Y[tr][:, c] == 1]
        pick += list(rng.choice(idx, size=min(a.per_class, len(idx)), replace=False))
    none_idx = tr[Y[tr].sum(1) == 0]
    pick += list(rng.choice(none_idx, size=min(a.per_class, len(none_idx)), replace=False))
    calib = to_tensor_input(X[np.array(pick)])                    # (N,3,H,W)
    np.save(p("_calib_nhwc.npy"), np.transpose(calib, (0, 2, 3, 1)).astype(np.float32))
    print(f"calibration wafers: {len(calib)}")

    # --- INT8 ---
    src = p(".onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(p(".onnx"), p("_pre.onnx")); src = p("_pre.onnx")
    except Exception as e:
        print("pre-process skipped:", e)
    quantize_static(src, p("_int8.onnx"), Reader(calib), quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, per_channel=True)

    # --- accuracy: thresholds tuned on the tune half, scored on the untouched test half ---
    P32_tu, P32_te = predict_onnx(p(".onnx"), Xv[tu]), predict_onnx(p(".onnx"), Xv[te])
    P8_tu, P8_te = predict_onnx(p("_int8.onnx"), Xv[tu]), predict_onnx(p("_int8.onnx"), Xv[te])
    Yt = Yv[te]
    thr32, thr8 = tune(P32_tu, Yv[tu]), tune(P8_tu, Yv[tu])
    m32 = metrics(P32_te, Yt, thr32)
    m8_same = metrics(P8_te, Yt, thr32)     # INT8 with the FP32 thresholds
    m8_tuned = metrics(P8_te, Yt, thr8)     # INT8 with thresholds re-tuned for INT8 (recommended)
    agree = float(((P32_te > thr32) == (P8_te > thr8)).all(1).mean())

    # --- size + latency (this machine's CPU; repeat on the Pi) ---
    x1 = to_tensor_input(Xv[te][:1])
    size_kb = lambda f: os.path.getsize(f) / 1024
    lat32, lat8 = latency_ms(p(".onnx"), x1), latency_ms(p("_int8.onnx"), x1)

    rows = {"FP32 ONNX": (m32, size_kb(p(".onnx")), lat32),
            "INT8 (FP32 thresholds)": (m8_same, size_kb(p("_int8.onnx")), lat8),
            "INT8 (re-tuned thresholds)": (m8_tuned, size_kb(p("_int8.onnx")), lat8)}
    table = pd.DataFrame({k: {"macro_f1": v[0]["macro_f1"], "false_alarm": v[0]["false_alarm"],
                              "miss": v[0]["miss"], "exact": v[0]["exact"],
                              "size_KB": v[1], "latency_ms": v[2]} for k, v in rows.items()}).T
    print("\nClean test half (%d wafers):" % len(te))
    print(table.round(3).to_string())
    print(f"\nFP32 and INT8 give identical label sets on {agree:.1%} of test wafers")
    pc = pd.DataFrame({"FP32": m32["per_class_f1"], "INT8 re-tuned": m8_tuned["per_class_f1"]}, index=PATTERNS)
    pc["change"] = pc["INT8 re-tuned"] - pc["FP32"]
    print("\nPer-class F1:"); print(pc.round(3).to_string())

    json.dump({"table": table.round(4).to_dict(orient="index"),
               "per_class_f1": pc.round(4).to_dict(orient="index"),
               "fp32_int8_agreement": agree, "calibration_wafers": len(calib),
               "thresholds_fp32": dict(zip(PATTERNS, thr32.tolist())),
               "thresholds_int8": dict(zip(PATTERNS, thr8.tolist()))},
              open(p("_export.json"), "w"), indent=2)
    print("\nSaved to", out_dir)
    print("For Hailo: use %s_b1.onnx and %s_calib_nhwc.npy" % (name, name))


if __name__ == "__main__":
    main()
