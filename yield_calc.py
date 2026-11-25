"""Die-level yield + spatial statistics: the 'domain credibility' layer.
All numbers are computed from the map itself, never invented by an LLM."""
import numpy as np


def die_stats(m):
    """m: 2D uint8 (0 outside, 1 pass, 2 fail)."""
    n = m.shape[0]
    y, x = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    cy, cx = (m.shape[0] - 1) / 2, (m.shape[1] - 1) / 2
    r = np.hypot(x - cx, y - cy) / (min(m.shape) / 2)
    on = m > 0
    fail = m == 2
    total, nfail = int(on.sum()), int(fail.sum())
    zones = {"center": (r < 0.35), "mid": (r >= 0.35) & (r < 0.75), "edge": (r >= 0.75)}
    zone_fail = {}
    for k, z in zones.items():
        zt = int((on & z).sum())
        zone_fail[k] = round(float((fail & z).sum() / zt), 4) if zt else 0.0
    return {
        "total_die": total, "failing_die": nfail,
        "die_yield": round(1 - nfail / total, 4) if total else 0.0,
        "zone_fail_rate": zone_fail,
    }


def radial_profile(m, bins=10):
    y, x = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    c = (m.shape[0] - 1) / 2
    r = np.hypot(x - c, y - c) / (m.shape[0] / 2)
    on, fail = m > 0, m == 2
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        z = on & (r >= a) & (r < b)
        out.append(float((fail & z).sum() / z.sum()) if z.sum() else 0.0)
    return out


def estimate_yield_impact(stats, baseline_yield=0.95, die_price=1.0):
    """Transparent formula: yield loss vs a baseline, in die and relative cost terms."""
    loss = max(0.0, baseline_yield - stats["die_yield"])
    return {
        "yield_loss_pts": round(loss * 100, 2),
        "lost_die_vs_baseline": int(round(loss * stats["total_die"])),
        "relative_cost_loss": round(loss * stats["total_die"] * die_price, 2),
        "assumption": f"baseline die yield {baseline_yield:.0%}; die price normalised to {die_price}",
    }


def lot_summary(maps_stats):
    """Aggregate per-wafer stats into a lot-level view (for the dashboard)."""
    ys = [s["die_yield"] for s in maps_stats]
    return {"wafers": len(ys), "mean_yield": round(float(np.mean(ys)), 4),
            "min_yield": round(float(np.min(ys)), 4), "std": round(float(np.std(ys)), 4)}
