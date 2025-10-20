from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from cycler import cycler

# Colorblind friendly palette (Okabe Ito)
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#999999",
]

SEED_COLOR = {
    1: PALETTE[1],  # orange
    2: PALETTE[0],  # blue
    3: PALETTE[2],  # green
    4: PALETTE[3],  # red
    5: PALETTE[4],  # magenta
}


def use_house_style() -> None:
    """Apply shared visualization defaults."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.loc": "best",
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2.0,
            "savefig.dpi": 150,
        }
    )


def format_thousands(ax) -> None:
    """Format both axes with thousands separators."""
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def color_for_seed(seed: int) -> str:
    """Return a stable color mapping for a given seed index."""
    return SEED_COLOR.get(int(seed), PALETTE[int(seed) % len(PALETTE)])


def finalize(ax=None, outfile: str | None = None) -> None:
    """Tighten layout and optionally persist the figure."""
    ax = ax or plt.gca()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
