"""Honest evaluation of a trained checkpoint.

The model never saw the validation lots during training. Here we split those
validation lots (by lot) into two halves: one half re-tunes the per-class
thresholds, the other half is a clean test set the thresholds never touched.

  python -m wafersentinel.evaluate --ckpt runs/wm811k.pt --pkl data/LSWMD_labelled.pkl

Needs the same --pkl and the default loader settings used for training, so the
80/20 lot split is reproduced exactly.
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix
from .classes import PATTERNS
from .data import load_wm811k, to_tensor_input
from .model import WaferCNN


def predict(model, X, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.tensor(to_tensor_input(X[i:i + bs]))
            out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out)


def tune(P, Y):
    thr = []
    for c in range(P.shape[1]):
        best = max(np.linspace(.1, .9, 17),
                   key=lambda t: f1_score(Y[:, c], P[:, c] > t, zero_division=0))
        thr.append(float(best))
    return np.array(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/wm811k.pt")
    ap.add_argument("--pkl", default="data/LSWMD_labelled.pkl")
    ap.add_argument("--out", default="runs/eval_report.json")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    model = WaferCNN(len(PATTERNS))
    model.load_state_dict(ck["state"])

    # Reproduce the exact training split, then take the held-out 20%.
    X, Y, G = load_wm811k(a.pkl, ck["size"], return_groups=True)
    _, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, Y, G))
    Xv, Yv, Gv = X[va], Y[va], G[va]

    # Split the held-out lots in half: tune thresholds on one half, test on the other.
    tu, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=1).split(Xv, Yv, Gv))
    thr = tune(predict(model, Xv[tu]), Yv[tu])
    Pe, Ye = predict(model, Xv[te]), Yv[te]
    pred = (Pe > thr).astype(int)

    print(f"tune wafers: {len(tu)} | test wafers: {len(te)}")
    print("test support per class:", dict(zip(PATTERNS, Ye.sum(0).astype(int))))
    print("thresholds:", dict(zip(PATTERNS, np.round(thr, 2))))

    p, r, f, s = precision_recall_fscore_support(Ye, pred, zero_division=0)
    table = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s.astype(int)}, index=PATTERNS)
    print("\nPer-class results on the clean test half:")
    print(table.round(3).to_string())
    print(f"\nMacro-F1 (patterns): {f.mean():.3f}")

    # 'none' (no defect) behaviour matters most in a real fab, where most wafers are clean.
    true_none = Ye.sum(1) == 0
    pred_none = pred.sum(1) == 0
    false_alarm = float((~pred_none[true_none]).mean()) if true_none.any() else float("nan")
    missed = float(pred_none[~true_none].mean()) if (~true_none).any() else float("nan")
    exact = float((pred == Ye).all(1).mean())
    print(f"Clean wafers flagged as defective (false alarm rate): {false_alarm:.3f}  [{int(true_none.sum())} clean wafers]")
    print(f"Defective wafers called clean (miss rate):            {missed:.3f}  [{int((~true_none).sum())} defective wafers]")
    print(f"Exact-match accuracy (all labels right):              {exact:.3f}")

    # 9x9 confusion matrix (8 patterns + none) on single-label / clean test wafers.
    labels = PATTERNS + ["none"]
    single = Ye.sum(1) <= 1
    yt = np.where(Ye[single].sum(1) == 0, 8, Ye[single].argmax(1))
    pm = Pe[single]
    yp = np.where((pm > thr).any(1), np.argmax(pm / thr, axis=1), 8)
    cm = confusion_matrix(yt, yp, labels=list(range(9)))
    cmdf = pd.DataFrame(cm, index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(cmdf.to_string())

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cmdf.to_csv(a.out.replace(".json", "_confusion.csv"))
    json.dump({
        "per_class": table.round(4).to_dict(orient="index"),
        "macro_f1": float(f.mean()),
        "false_alarm_rate": false_alarm, "miss_rate": missed, "exact_match": exact,
        "thresholds": dict(zip(PATTERNS, thr.tolist())),
        "n_tune": int(len(tu)), "n_test": int(len(te)),
    }, open(a.out, "w"), indent=2)
    print(f"\nSaved {a.out} and {a.out.replace('.json', '_confusion.csv')}")


if __name__ == "__main__":
    main()
