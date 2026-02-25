## helper script for all the plotting functions
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr

def violin_two_groups_by_bool_mean_matplotlib(
    df,
    val_col,
    bool_col,
    ax=None,
    order=(False, True),
    group_labels=None,

    # layout
    x_gap=0.55,
    width=0.35,
    fig_size=(4.6, 4.4),

    # style
    colors=("#3a923a", "#8e6bbf"),
    grid_color="0.9",
    linewidth=1.1,
    alpha=1.0,
    font_family="Arial",

    # labels
    x_label="",
    y_label=None,
    title=None,
    y_tick_step=None,


    # counts
    show_n=True,
    n_fmt="n={:,}",

    # ---- NEW: classic inner box style ----
    show_box=True,             # draw inner box (IQR) + median line + whiskers
    box_width_frac=0.25,
    box_facecolor="black",   # <-- solid black IQR box
    box_alpha=1.0,
    box_edgecolor="black",
    box_lw=1.2,
    whisker_lw=1.2,
    cap_lw=1.2,
    median_lw=2.8,           # <-- thicker median line


    # ---- NEW: mean dot optional ----
    show_mean=False,
    mean_dot_size=120,
    mean_dot_face="white",
    mean_dot_edge="black",
    mean_dot_lw=1.2,

    # stats
    do_mannwhitney=True,
    mannwhitney_alternative="two-sided",
    other_pvals=None,
    stats_fontsize=11,

    # SVG
    rasterize_violins=False,
    save_svg=None,
    dpi=800,
):
    """
    Two-group violin plot split by boolean column.

    Style:
      - matplotlib.violinplot for colored violins
      - optional matplotlib.boxplot overlay for classic "inner box" (IQR + median + whiskers)
      - optional mean dot
    """

    # ---- rc for SVG + font ----
    prev_font = mpl.rcParams.get("font.family", None)
    prev_svg  = mpl.rcParams.get("svg.fonttype", None)
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["svg.fonttype"] = "none"

    # ---- clean data ----
    d = df[[val_col, bool_col]].dropna().copy()
    d[bool_col] = d[bool_col].astype(bool)

    a_bool, b_bool = bool(order[0]), bool(order[1])

    if group_labels is None:
        lab_a, lab_b = str(a_bool), str(b_bool)
    elif isinstance(group_labels, dict):
        lab_a = group_labels.get(a_bool, str(a_bool))
        lab_b = group_labels.get(b_bool, str(b_bool))
    else:
        lab_a, lab_b = group_labels

    vals_a = d.loc[d[bool_col] == a_bool, val_col].to_numpy(float)
    vals_b = d.loc[d[bool_col] == b_bool, val_col].to_numpy(float)
    vals_a = vals_a[np.isfinite(vals_a)]
    vals_b = vals_b[np.isfinite(vals_b)]

    if vals_a.size == 0 or vals_b.size == 0:
        raise ValueError("One group empty after filtering.")

    # ---- stats ----
    U = pval = np.nan
    if do_mannwhitney:
        res = mannwhitneyu(vals_a, vals_b, alternative=mannwhitney_alternative)
        U = float(res.statistic)
        pval = float(res.pvalue)

    qval = None
    if other_pvals is not None and np.isfinite(pval):
        _, qvec, _, _ = multipletests([pval] + list(other_pvals), method="fdr_bh")
        qval = float(qvec[0])

    # ---- fig/ax ----
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    else:
        fig = ax.figure

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid_color, lw=1)

    # ---- violins ----
    positions = [0.0, x_gap]
    parts = ax.violinplot(
        [vals_a, vals_b],
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor("black")
        body.set_linewidth(linewidth)
        body.set_alpha(alpha)
        if rasterize_violins:
            body.set_rasterized(True)

    # ---- inner classic box (IQR + median + whiskers) ----
    if show_box:
        box_width = width * box_width_frac
        bp = ax.boxplot(
            [vals_a, vals_b],
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            zorder=4,
        )

        # boxes
        # boxes (solid black)
        for patch in bp["boxes"]:
            patch.set_facecolor(box_facecolor)
            patch.set_alpha(box_alpha)
            patch.set_edgecolor(box_edgecolor)
            patch.set_linewidth(box_lw)

        # whiskers
        for w in bp["whiskers"]:
            w.set_color("black")
            w.set_linewidth(whisker_lw)

        # caps
        for c in bp["caps"]:
            c.set_color("black")
            c.set_linewidth(cap_lw)

        # median (white + thick)
        for m in bp["medians"]:
            m.set_color("white")
            m.set_linewidth(median_lw)

    # ---- optional mean dot ----
    if show_mean:
        means = [vals_a.mean(), vals_b.mean()]
        ax.scatter(
            positions,
            means,
            s=mean_dot_size,
            color=mean_dot_face,
            edgecolor=mean_dot_edge,
            linewidth=mean_dot_lw,
            zorder=5,
        )

    # ---- x tick labels + counts ----
    n_a, n_b = len(vals_a), len(vals_b)
    if show_n:
        xt = [
            f"{lab_a}\n({n_fmt.format(n_a)})",
            f"{lab_b}\n({n_fmt.format(n_b)})",
        ]
    else:
        xt = [lab_a, lab_b]

    ax.set_xticks(positions)
    ax.set_xticklabels(xt)

    # ---- labels/title ----
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label if y_label else val_col)
    if title:
        ax.set_title(title)

    # ---- y-lims + stats text ----
    ymin = np.nanmin(np.concatenate([vals_a, vals_b]))
    ymax = np.nanmax(np.concatenate([vals_a, vals_b]))
    yr = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.05 * yr, ymax + 0.18 * yr)
    
    if y_tick_step is not None:
        ymin, ymax = ax.get_ylim()
        yticks = np.arange(
            np.floor(ymin / y_tick_step) * y_tick_step,
            np.ceil(ymax / y_tick_step) * y_tick_step + y_tick_step,
            y_tick_step,
        )
        ax.set_yticks(yticks)


    if do_mannwhitney and np.isfinite(pval):
        stat_txt = (
            f"MWU U={U:.2g}, p={pval:.2g}"
            if qval is None else f"MWU U={U:.2g}, q={qval:.2g}"
        )
        ax.text(
            np.mean(positions),
            ymax + 0.06 * yr,
            stat_txt,
            ha="center",
            va="bottom",
            fontsize=stats_fontsize,
        )

    ax.set_xlim(-0.5, x_gap + 0.5)
    ax.tick_params(
    axis="both",
    which="major",
    direction="out",     # outward ticks (journal style)
    bottom=True,
    top=False,
    left=True,
    right=False,
    length=6,            # tick length
    width=1.2,           # tick thickness
)

    fig.tight_layout()

    # ---- save ----
    if save_svg:
        if os.path.dirname(save_svg):
            os.makedirs(os.path.dirname(save_svg), exist_ok=True)
        fig.savefig(save_svg, format="svg", dpi=dpi, bbox_inches="tight", transparent=True)

    # restore rc
    if prev_font is not None:
        mpl.rcParams["font.family"] = prev_font
    if prev_svg is not None:
        mpl.rcParams["svg.fonttype"] = prev_svg

    stats = {"U": U, "p": pval, "q": qval}
    return fig, ax, d, stats




def scatter_matrix_entries(
    M1,
    M2,
    *,
    xlabel="Matrix 1",
    ylabel="Matrix 2",
    title=None,
    s=8,
    alpha=0.4,
    color="#1f77b4",
    fit_line=True,
    rasterize_points=True,
    save_svg=None,          # NEW: path to save SVG
    dpi=1200,
):
    """
    Scatter plot of corresponding entries in two same-shaped matrices
    and compute Pearson correlation.

    Parameters
    ----------
    M1, M2 : array-like
        Matrices of identical shape.
    rasterize_points : bool
        Rasterize scatter points (recommended for large matrices).
    save_svg : str or None
        If provided, save figure as SVG with rasterized points.
    dpi : int
        DPI used for rasterized elements in SVG.

    Returns
    -------
    r, p : float
        Pearson correlation coefficient and p-value.
    """

    # --- SVG-friendly text ---
    prev_svg = mpl.rcParams.get("svg.fonttype", None)
    mpl.rcParams["svg.fonttype"] = "none"

    M1 = np.asarray(M1)
    M2 = np.asarray(M2)

    if M1.shape != M2.shape:
        raise ValueError("M1 and M2 must have the same shape")

    # flatten + drop NaNs / infs
    x = M1.ravel()
    y = M2.ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Not enough valid entries to compute correlation")

    # Pearson correlation
    r, p = pearsonr(x, y)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5, 5))

    sc = ax.scatter(x, y, s=s, alpha=alpha, color=color)
    if rasterize_points:
        sc.set_rasterized(True)

    if fit_line:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, m * xx + b, color="black", lw=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.text(
        0.02, 0.98,
        f"Pearson r = {r:.3f}\n"
        f"p = {p:.2e}\n"
        f"N = {len(x)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.grid(False)
    plt.tight_layout()

    # --- save SVG ---
    if save_svg:
        fig.savefig(
            save_svg,
            format="svg",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
        )

    plt.show()

    # restore rcParams
    if prev_svg is not None:
        mpl.rcParams["svg.fonttype"] = prev_svg

    return r, p

