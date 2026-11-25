"""Loaders for the real datasets.

WM-811K  : LSWMD.pkl (Kaggle 'qingyi/wm811k-wafer-map'). Columns: waferMap (2D uint8,
           variable size), failureType (e.g. [['Scratch']]).
MixedWM38: Wafer_Map_Datasets.npz from the Junliangwangdhu/WaferMap GitHub repo;
           'arr_0' = maps (52x52, 0/1/2), 'arr_1' = 38-class one-hot.
"""
import numpy as np
from scipy.ndimage import zoom
from .classes import P2I, NUM_CLASSES, WM811K_MAP


def resize_map(m, n=64):
    """Nearest-neighbour resize so the 0/1/2 encoding stays discrete."""
    return zoom(m, (n / m.shape[0], n / m.shape[1]), order=0).astype(np.uint8)


def load_wm811k(pkl_path, n=64, max_none=8000, seed=0, return_groups=False):
    import pandas as pd
    df = pd.read_pickle(pkl_path)
    df["ft"] = df["failureType"].apply(lambda a: a[0][0])
    # random (not first-N) subsample of the huge 'none' class
    none_idx = df.index[df["ft"] == "none"].to_numpy()
    keep = np.random.default_rng(seed).choice(none_idx, size=min(max_none, len(none_idx)), replace=False)
    df = df[(df["ft"] != "none") | df.index.isin(keep)]
    X, Y, G = [], [], []
    for m, ft, lot in zip(df["waferMap"], df["ft"], df["lotName"]):
        lab = WM811K_MAP.get(ft)
        y = np.zeros(NUM_CLASSES, np.float32)
        if lab is not None:
            y[P2I[lab]] = 1
        if hasattr(lot, "__len__") and not isinstance(lot, str):
            lot = lot[0]
        X.append(resize_map(m, n)); Y.append(y); G.append(str(lot))
    X, Y = np.stack(X), np.stack(Y)
    return (X, Y, np.array(G)) if return_groups else (X, Y)


def mixedwm38_label_table():
    """38-class index -> basic patterns. VERIFY the order against the dataset README for
    your copy; singles shown here, fill mixtures (typically 2-4 patterns each)."""
    singles = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]
    table = {0: []}
    for i, s in enumerate(singles, 1):
        table[i] = [s]
    return table


def load_mixedwm38(npz_path, n=64):
    d = np.load(npz_path)
    table = mixedwm38_label_table()
    X, Y = [], []
    for m, oh in zip(d["arr_0"], d["arr_1"]):
        k = int(np.argmax(oh))
        if k not in table:
            continue
        y = np.zeros(NUM_CLASSES, np.float32)
        for p in table[k]:
            y[P2I[p]] = 1
        X.append(resize_map(m, n)); Y.append(y)
    return np.stack(X), np.stack(Y)


def to_tensor_input(maps):
    """uint8 (N,H,W) with 0/1/2 -> float32 (N,3,H,W) one-hot [outside, pass, fail].
    One-hot beats raw 0/1/2 because the values are categorical."""
    maps = np.asarray(maps)
    return np.stack([(maps == k) for k in (0, 1, 2)], axis=1).astype(np.float32)
