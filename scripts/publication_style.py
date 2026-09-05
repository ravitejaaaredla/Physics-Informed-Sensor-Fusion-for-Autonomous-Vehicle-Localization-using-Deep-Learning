import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# PUBLICATION FIGURE STYLE
#
# Designed for thesis / journal / conference figures.
# Minimal colour, clean white background, vector-friendly.
# ============================================================

GT_COLOR = "#0057B8"      # restrained academic blue
PRED_COLOR = "#000000"    # black
GRAY_DARK = "#555555"
GRAY_MID = "#888888"
GRAY_LIGHT = "#D9D9D9"


def apply_publication_style():

    mpl.rcParams.update({

        # ----------------------------------------------------
        # Typography
        # ----------------------------------------------------

        "font.family": "serif",

        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],

        "font.size": 9,

        "axes.titlesize": 9,

        "axes.labelsize": 9,

        "xtick.labelsize": 8,

        "ytick.labelsize": 8,

        "legend.fontsize": 8,


        # ----------------------------------------------------
        # Figure
        # ----------------------------------------------------

        "figure.facecolor": "white",

        "savefig.facecolor": "white",

        "savefig.transparent": False,

        "figure.dpi": 120,

        "savefig.dpi": 600,


        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        "axes.facecolor": "white",

        "axes.edgecolor": "black",

        "axes.linewidth": 0.8,

        "axes.grid": False,

        "axes.spines.top": False,

        "axes.spines.right": False,

        "axes.titlepad": 6,


        # ----------------------------------------------------
        # Ticks
        # ----------------------------------------------------

        "xtick.direction": "out",

        "ytick.direction": "out",

        "xtick.major.size": 3.5,

        "ytick.major.size": 3.5,

        "xtick.major.width": 0.8,

        "ytick.major.width": 0.8,


        # ----------------------------------------------------
        # Lines
        # ----------------------------------------------------

        "lines.linewidth": 1.4,

        "lines.markersize": 4,


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        "legend.frameon": False,

        "legend.borderaxespad": 0.4,

        "legend.handlelength": 2.0,


        # ----------------------------------------------------
        # PDF / SVG text
        #
        # Keeps text editable/vector where possible.
        # ----------------------------------------------------

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

        "svg.fonttype": "none",
    })


def clean_axis(ax):

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        width=0.8,
        length=3.5,
    )

    ax.set_axisbelow(True)


def subtle_grid(
    ax,
    axis="both",
):

    ax.grid(
        True,
        which="major",
        axis=axis,
        linestyle=":",
        linewidth=0.45,
        color="#BFBFBF",
        alpha=0.55,
    )

    ax.set_axisbelow(True)


def panel_label(
    ax,
    label,
):

    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_publication_figure(
    fig,
    output_stem,
):

    # Vector master copy for publication.
    fig.savefig(
        str(output_stem) + ".pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )

    # SVG useful for editing in Illustrator/Inkscape.
    fig.savefig(
        str(output_stem) + ".svg",
        bbox_inches="tight",
        pad_inches=0.03,
    )

    # High-resolution raster copy for Word/thesis.
    fig.savefig(
        str(output_stem) + ".png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )


if __name__ == "__main__":

    apply_publication_style()

    print("=" * 70)
    print("PUBLICATION STYLE CREATED")
    print("=" * 70)
    print("Font            : Times New Roman / serif fallback")
    print("Ground Truth    : Blue")
    print("Model Prediction: Black")
    print("Other series    : Grayscale")
    print("Background      : White")
    print("Titles          : Minimal / captions belong in document")
    print("Grid            : Off by default")
    print("PNG             : 600 dpi")
    print("PDF             : Vector")
    print("SVG             : Vector/editable")
    print("=" * 70)
