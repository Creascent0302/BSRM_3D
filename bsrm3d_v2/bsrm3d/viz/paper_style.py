"""
Paper-quality plot styling for BSRM-3D.

All figures: Times New Roman, vectorized PDF output, consistent colors.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global style ──────────────────────────────────────────────────────
def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })

# ── Colors ────────────────────────────────────────────────────────────
COLORS = {
    "BSRM3D":      "#d62728",   # red
    "delta-PRM":   "#1f77b4",   # blue
    "PRM*":        "#9467bd",   # purple
    "Halton-PRM":  "#2ca02c",   # green
    "SPARS2":      "#ff7f0e",   # orange
    "RRT":         "#8c564b",   # brown
    "RRT-Connect": "#e377c2",   # pink
}

MARKERS = {
    "BSRM3D": "o", "delta-PRM": "s", "PRM*": "D",
    "Halton-PRM": "^", "SPARS2": "v", "RRT": "x", "RRT-Connect": "+",
}

PLANNER_ORDER = ["BSRM3D", "SPARS2", "delta-PRM", "PRM*", "Halton-PRM"]

# ── Helpers ───────────────────────────────────────────────────────────
def save_fig(fig, path):
    """Save as both PDF (vector) and PNG (preview)."""
    fig.savefig(path.replace(".png", ".pdf"), format="pdf")
    fig.savefig(path, format="png")
    plt.close(fig)
