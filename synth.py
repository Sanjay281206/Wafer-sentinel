"""Synthetic wafer-map generator: smoke-test the pipeline before downloading
WM-811K / MixedWM38, and augment rare classes (Donut, Near-full).
Encoding follows WM-811K: 0 = outside wafer, 1 = passing die, 2 = failing die."""
import numpy as np
from .classes import PATTERNS, P2I, NUM_CLASSES


def _grid(n):
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    r = np.hypot(x - c, y - c) / (n / 2)
    th = np.arctan2(y - c, x - c)
    return x, y, r, th


def make_map(patterns, n=64, rng=None, base_fail=0.02):
    rng = rng or np.random.default_rng()
    x, y, r, th = _grid(n)
    wafer = r <= 1.0
    p = np.full((n, n), base_fail)
    for pat in patterns:
        if pat == "Center":
            p += 0.85 * np.exp(-(r / rng.uniform(0.15, 0.3)) ** 2)
        elif pat == "Donut":
            p += 0.8 * np.exp(-((r - rng.uniform(0.35, 0.6)) / 0.1) ** 2)
        elif pat == "Edge-Ring":
            p += 0.85 * (r > rng.uniform(0.82, 0.92))
        elif pat == "Edge-Loc":
            a = rng.uniform(-np.pi, np.pi)
            d = np.angle(np.exp(1j * (th - a)))
            p += 0.85 * ((r > 0.75) & (np.abs(d) < rng.uniform(0.3, 0.6)))
        elif pat == "Loc":
            cx = n / 2 + rng.uniform(0.15, 0.6) * n * rng.choice([-1, 1]) / 1.5
            cy = n / 2 + rng.uniform(0.15, 0.6) * n * rng.choice([-1, 1]) / 1.5
            p += 0.85 * (np.hypot(x - cx, y - cy) < rng.uniform(0.07, 0.13) * n)
        elif pat == "Near-full":
            p += rng.uniform(0.7, 0.9)
        elif pat == "Scratch":
            a = rng.uniform(0, np.pi)
            c0 = rng.uniform(-0.4, 0.4) * n + n / 2
            dist = np.abs((x - c0) * np.sin(a) - (y - n / 2) * np.cos(a))
            p += 0.9 * (dist < 1.2)
        elif pat == "Random":
            p += rng.uniform(0.1, 0.25)
    fail = rng.random((n, n)) < np.clip(p, 0, 0.98)
    out = np.zeros((n, n), np.uint8)
    out[wafer] = 1
    out[wafer & fail] = 2
    return out


def make_dataset(n_samples=2000, n=64, p_none=0.3, p_mixed=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n, n), np.uint8)
    Y = np.zeros((n_samples, NUM_CLASSES), np.float32)
    for i in range(n_samples):
        pats = [] if rng.random() < p_none else list(
            rng.choice(PATTERNS, size=2 if rng.random() < p_mixed else 1, replace=False))
        X[i] = make_map(pats, n, rng)
        for pt in pats:
            Y[i, P2I[pt]] = 1
    return X, Y
