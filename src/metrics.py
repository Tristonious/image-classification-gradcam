import numpy as np


def gini_coefficient(x):
    """
    Gini coefficient of a non-negative array.
    0 = perfectly uniform, 1 = all mass in one pixel.
    """
    x = np.ravel(x).astype(np.float64)
    if np.all(x == 0):
        return 0.0
    x = np.maximum(x, 0)
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1)
    sum_x = x_sorted.sum()
    g = (1.0 / (n - 1.0)) * (
        (n + 1.0) - 2.0 * (np.sum((n + 1.0 - index) * x_sorted) / sum_x)
    )
    return float(g)


def compute_heatmap_metrics(heatmap):
    """
    Quantify Grad-CAM heatmap focus and distribution.

    Returns a dict with:
      entropy_bits   — Shannon entropy (bits); lower = more focused
      entropy_norm   — entropy normalized to [0, 1] relative to uniform
      focus_top1pct  — fraction of activation mass in top 1% of pixels
      focus_top5pct  — fraction of activation mass in top 5% of pixels
      focus_top10pct — fraction of activation mass in top 10% of pixels
      gini           — Gini coefficient of activation distribution
      mean, std      — basic stats of the normalized probability map
    """
    h = np.asarray(heatmap, dtype=np.float64)
    h = np.maximum(h, 0)
    s = h.sum()
    n = h.size
    p = (h / s).ravel() if s > 0 else np.full(n, 1.0 / n)

    eps = 1e-12
    entropy_bits = -np.sum(p * np.log2(p + eps))
    entropy_norm = entropy_bits / np.log2(n)

    def focus_top_pct(p, pct):
        k = max(1, int(pct * p.size))
        idx = np.argpartition(p, -k)[-k:]
        return float(p[idx].sum())

    return {
        "entropy_bits":   float(entropy_bits),
        "entropy_norm":   float(entropy_norm),
        "focus_top1pct":  focus_top_pct(p, 0.01),
        "focus_top5pct":  focus_top_pct(p, 0.05),
        "focus_top10pct": focus_top_pct(p, 0.10),
        "gini":           gini_coefficient(p),
        "mean":           float(p.mean()),
        "std":            float(p.std()),
    }
