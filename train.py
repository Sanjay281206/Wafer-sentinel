"""Train the multi-label CNN.
  python -m wafersentinel.train --source synth --epochs 15
  python -m wafersentinel.train --source wm811k --pkl data/LSWMD_labelled.pkl
  python -m wafersentinel.train --source mixed  --npz data/Wafer_Map_Datasets.npz
"""
import argparse, json, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from .classes import PATTERNS
from .data import load_wm811k, load_mixedwm38, to_tensor_input
from .synth import make_dataset
from .model import WaferCNN, FocalBCE


def get_data(a):
    """Returns X, Y and a group id per sample (lot name for WM-811K) so the
    train/val split never puts wafers from the same lot on both sides."""
    if a.source == "synth":
        X, Y = make_dataset(a.n_synth, a.size)
        return X, Y, np.arange(len(X))
    if a.source == "wm811k":
        return load_wm811k(a.pkl, a.size, return_groups=True)
    X, Y = load_mixedwm38(a.npz, a.size)
    return X, Y, np.arange(len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synth", choices=["synth", "wm811k", "mixed"])
    ap.add_argument("--pkl"); ap.add_argument("--npz")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n_synth", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/wafercnn.pt")
    a = ap.parse_args()

    X, Y, G = get_data(a)
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, Y, G))
    Xtr, Xva, Ytr, Yva = X[tr], X[va], Y[tr], Y[va]
    print("train", len(tr), "val", len(va), "| val class counts:", dict(zip(PATTERNS, Yva.sum(0).astype(int))))

    xt = torch.tensor(to_tensor_input(Xtr)); yt = torch.tensor(Ytr)
    xv = torch.tensor(to_tensor_input(Xva)); yv = torch.tensor(Yva)
    pos = yt.sum(0).clamp(min=1)
    pw = ((len(yt) - pos) / pos).clamp(max=20.0)       # per-class positive weight
    model = WaferCNN(len(PATTERNS))
    crit = FocalBCE(2.0, pw)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * ((len(xt) + a.bs - 1) // a.bs))
    dl = DataLoader(TensorDataset(xt, yt), a.bs, shuffle=True)
    for ep in range(a.epochs):
        model.train(); tot = 0
        for xb, yb in dl:
            # cheap augmentation: random flips + 90deg rotations (wafer patterns are rotation-invariant)
            k = np.random.randint(4); xb = torch.rot90(xb, k, (2, 3))
            if np.random.rand() < .5: xb = xb.flip(3)
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval()
        with torch.no_grad(): pv = (torch.sigmoid(model(xv)) > .5).int().numpy()
        f1 = f1_score(yv.numpy(), pv, average="macro", zero_division=0)
        print(f"epoch {ep+1:02d} loss {tot/len(dl):.4f} macroF1 {f1:.3f}")
    # per-class threshold tuning on the validation set
    with torch.no_grad(): pr = torch.sigmoid(model(xv)).numpy()
    thr = []
    for c in range(len(PATTERNS)):
        best = max(np.linspace(.1, .9, 17), key=lambda t: f1_score(yv[:, c], pr[:, c] > t, zero_division=0))
        thr.append(float(best))
    print({p: round(f1_score(yv[:, i], pr[:, i] > thr[i], zero_division=0), 3) for i, p in enumerate(PATTERNS)})
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict(), "thresholds": thr, "size": a.size}, a.out)
    json.dump(thr, open(a.out.replace(".pt", "_thr.json"), "w"))


if __name__ == "__main__":
    main()