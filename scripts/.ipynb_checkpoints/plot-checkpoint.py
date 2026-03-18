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


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
def violin_two_groups_by_bool_median_matplotlib(
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
    ylim=None,

    # labels
    x_label="",
    y_label=None,
    title=None,

    # counts
    show_n=True,
    n_fmt="n={:,}",

    # ---- classic inner box style ----
    show_box=True,
    box_width_frac=0.25,
    box_facecolor="black",
    box_alpha=1.0,
    box_edgecolor="black",
    box_lw=1.2,
    whisker_lw=1.2,
    cap_lw=1.2,
    median_lw=2.8,

    # ---- mean dot optional ----
    show_mean=False,
    mean_dot_size=120,
    mean_dot_face="white",
    mean_dot_edge="black",
    mean_dot_lw=1.2,

    # ---- value transforms ----
    transform="none",           # "none" | "log2" | "log2p1" | "log2_signed" | "zscore" | "robust_zscore"
    log_eps=1e-9,               # used for log transforms
    zscore_scope="pooled",      # "pooled" | "within_group"

    # ---- NEW: annotate median text above each violin ----
    annotate_median=True,
    median_fmt="med={:.2f}",
    median_text_color="black",
    median_text_size=10,
    median_text_weight="bold",
    median_text_yfrac=0.965,    # relative (0..1) within axis y-lims
    median_text_box=True,
    median_text_box_fc="white",
    median_text_box_ec="none",
    median_text_box_alpha=0.75,

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

    Transforms:
      - "log2": allows negatives by shifting all values by (-min + eps) if needed, then log2(x + shift)
      - "log2_signed": sign(x) * log2(1 + |x|)
      - "log2p1": log2(1 + x), requires x>=0
      - "zscore"/"robust_zscore": pooled vs within-group scaling

    Median text annotation (annotate_median=True):
      Writes median above each violin using final axis limits (works with custom ylim).
    """

    # ---- rc for SVG + font ----
    prev_font = mpl.rcParams.get("font.family", None)
    prev_svg = mpl.rcParams.get("svg.fonttype", None)
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

    # ----------------------------
    # transform values
    # ----------------------------
    transform = (transform or "none").lower()
    zscore_scope = (zscore_scope or "pooled").lower()

    def _zscore(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
        sd = float(sd)
        if not np.isfinite(sd) or sd <= 0:
            return np.zeros_like(x, dtype=float)
        return (x - float(mu)) / sd

    def _robust_zscore(x: np.ndarray, med: float, mad: float) -> np.ndarray:
        denom = 1.4826 * float(mad)
        if not np.isfinite(denom) or denom <= 0:
            return np.zeros_like(x, dtype=float)
        return (x - float(med)) / denom

    if transform == "none":
        pass

    elif transform in ("log2", "log"):
        pooled = np.concatenate([vals_a, vals_b])
        min_val = float(np.min(pooled))
        # shift so argument is >0
        shift = (-min_val + float(log_eps)) if min_val <= 0 else float(log_eps)
        vals_a = np.log2(vals_a + shift)
        vals_b = np.log2(vals_b + shift)

    elif transform in ("log2_signed", "signed_log2"):
        vals_a = np.sign(vals_a) * np.log2(1.0 + np.abs(vals_a))
        vals_b = np.sign(vals_b) * np.log2(1.0 + np.abs(vals_b))

    elif transform == "log2p1":
        pooled = np.concatenate([vals_a, vals_b])
        if np.any(pooled < 0):
            raise ValueError("transform='log2p1' requires non-negative values.")
        vals_a = np.log2(1.0 + vals_a)
        vals_b = np.log2(1.0 + vals_b)

    elif transform in ("zscore", "z-score", "zs"):
        if zscore_scope == "pooled":
            pooled = np.concatenate([vals_a, vals_b])
            mu = np.mean(pooled)
            sd = np.std(pooled, ddof=0)
            vals_a = _zscore(vals_a, mu, sd)
            vals_b = _zscore(vals_b, mu, sd)
        elif zscore_scope == "within_group":
            vals_a = _zscore(vals_a, np.mean(vals_a), np.std(vals_a, ddof=0))
            vals_b = _zscore(vals_b, np.mean(vals_b), np.std(vals_b, ddof=0))
        else:
            raise ValueError("zscore_scope must be 'pooled' or 'within_group'.")

    elif transform in ("robust_zscore", "robust", "mad"):
        def mad1(x):
            med = np.median(x)
            return med, np.median(np.abs(x - med))

        if zscore_scope == "pooled":
            pooled = np.concatenate([vals_a, vals_b])
            med, madv = mad1(pooled)
            vals_a = _robust_zscore(vals_a, med, madv)
            vals_b = _robust_zscore(vals_b, med, madv)
        elif zscore_scope == "within_group":
            med_a, mad_a = mad1(vals_a)
            med_b, mad_b = mad1(vals_b)
            vals_a = _robust_zscore(vals_a, med_a, mad_a)
            vals_b = _robust_zscore(vals_b, med_b, mad_b)
        else:
            raise ValueError("zscore_scope must be 'pooled' or 'within_group'.")
    else:
        raise ValueError(
            "transform must be one of: 'none', 'log2', 'log2p1', 'log2_signed', 'zscore', 'robust_zscore'."
        )

    # ---- stats (computed on transformed values) ----
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
        for patch in bp["boxes"]:
            patch.set_facecolor(box_facecolor)
            patch.set_alpha(box_alpha)
            patch.set_edgecolor(box_edgecolor)
            patch.set_linewidth(box_lw)
        for w in bp["whiskers"]:
            w.set_color("black")
            w.set_linewidth(whisker_lw)
        for c in bp["caps"]:
            c.set_color("black")
            c.set_linewidth(cap_lw)
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
    if y_label is None:
        if transform == "none":
            ylab = val_col
        elif transform == "log2":
            ylab = f"log2({val_col}+shift)"
        elif transform == "log2_signed":
            ylab = f"signed log2(1+|{val_col}|)"
        elif transform == "log2p1":
            ylab = f"log2(1+{val_col})"
        elif transform == "zscore":
            ylab = f"{val_col} (z-score)"
        else:
            ylab = f"{val_col} (robust z-score)"
        ax.set_ylabel(ylab)
    else:
        ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    # ---- y-lims ----
    data_ymin = float(np.nanmin(np.concatenate([vals_a, vals_b])))
    data_ymax = float(np.nanmax(np.concatenate([vals_a, vals_b])))
    data_yr = data_ymax - data_ymin if data_ymax > data_ymin else 1.0

    if ylim is None:
        ax.set_ylim(data_ymin - 0.05 * data_yr, data_ymax + 0.18 * data_yr)
    else:
        if (not isinstance(ylim, (tuple, list))) or len(ylim) != 2:
            raise ValueError("ylim must be None or a (ymin, ymax) tuple.")
        y0, y1 = float(ylim[0]), float(ylim[1])
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
            raise ValueError(f"Invalid ylim={ylim}. Must be finite and ymax > ymin.")
        ax.set_ylim(y0, y1)

    # ---- NEW: median text annotations (uses final axis limits) ----
    # ---- median annotation (colored to match violin) ----
    if annotate_median:
        meds = [float(np.median(vals_a)), float(np.median(vals_b))]

        # Use x in data coords, y in axes coords (prevents clipping)
        trans = ax.get_xaxis_transform()

        for x, med, col in zip(positions, meds, colors):
            ax.text(
                x,
                median_text_yfrac,  # e.g. 0.965
                f"median = {med:.3f}",
                transform=trans,
                ha="center",
                va="top",
                fontsize=median_text_size,
                fontweight=median_text_weight,
                color=col,              # match violin color
                clip_on=False,
                zorder=6,
            )


    # ---- stats text (also uses final axis limits) ----
    if do_mannwhitney and np.isfinite(pval):
        stat_txt = (
            f"MWU U={U:.2g}, p={pval:.2g}"
            if qval is None else f"MWU U={U:.2g}, q={qval:.2g}"
        )
        y0, y1 = ax.get_ylim()
        y_stat = y0 + 0.92 * (y1 - y0)
        ax.text(
            np.mean(positions),
            y_stat,
            stat_txt,
            ha="center",
            va="top",
            fontsize=stats_fontsize,
        )

    ax.set_xlim(-0.5, x_gap + 0.5)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        bottom=True,
        top=False,
        left=True,
        right=False,
        length=6,
        width=1.2,
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


from matplotlib.ticker import PercentFormatter
import pandas as pd
def plot_bars_with_fisher_annotations(
    df_props,                 # must contain ['gene','p_chr7','p_ecDNA','pval'] ; 'delta' optional
    df_pb=None,               # needed only when plot_exon=True; columns ['gene','exon_expr'] (CPM)
    sort_by="p_ecDNA",
    figsize=(14, 6),
    rotate_xticks=45,
    plot_exon=True,
    title=None,
    save_svg_path=None,       # <-- NEW: if given, saves SVG there
    dpi=200,                   # irrelevant for pure SVG vectors, but kept for consistency,
    color_ec = 'C0',
    color_chr7 = 'C1'
):
    # --- set Arial font locally, restore after ---
    old_rc = plt.rcParams.copy()
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["svg.fonttype"] = "none"  # keep text as text in SVG (not paths)

    try:
        def _p_to_stars(p):
            if not np.isfinite(p): return "n.s."
            if p < 1e-4: return "***"
            if p < 1e-3: return "**"
            if p < 1e-2: return "*"
            return "n.s."

        # ensure delta exists
        if "delta" not in df_props.columns:
            df_props = df_props.assign(delta=df_props["p_ecDNA"] - df_props["p_chr7"])

        # merge pseudobulk (only if plotting it)
        if plot_exon:
            if df_pb is None:
                raise ValueError("df_pb is required when plot_exon=True")
            df_pb2 = df_pb.copy()
            df_pb2["logCPM"] = np.log10(np.clip(df_pb2["exon_expr"].astype(float), 1e-6, None))
            dfm = (df_props.merge(df_pb2[["gene","logCPM"]], on="gene", how="left")
                          .dropna(subset=["p_chr7","p_ecDNA","logCPM"]))
        else:
            dfm = df_props.dropna(subset=["p_chr7","p_ecDNA"]).copy()

        # sort
        dfm = dfm.sort_values(sort_by, ascending=False, ignore_index=True)

        genes = dfm["gene"].tolist()
        x = np.arange(len(genes))
        width = 0.38

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # ecDNA (left), chr7 (right)
        bars_ec   = ax.bar(x - width/2, dfm["p_ecDNA"].to_numpy(float), width, color=color_ec, label="ecDNA")
        bars_chr7 = ax.bar(x + width/2, dfm["p_chr7"].to_numpy(float),  width, color=color_chr7, label="chr7")

        # optionally add pseudobulk logCPM as negative green bars with right axis
        if plot_exon:
            log_min = float(dfm["logCPM"].min()); log_max = float(dfm["logCPM"].max())
            pad = 0.05 * max(log_max - log_min, 1e-9)
            lo, hi = log_min - pad, log_max + pad
            rng = hi - lo
            t = np.clip((dfm["logCPM"] - lo) / rng, 0, 1)   # 0..1 after padding
            neg_heights = -t.values
            ax.bar(x, neg_heights, width=width*0.9, color="green", alpha=0.65, label="Exon expr (logCPM)")

        # x labels
        ax.set_xticks(x)
        ax.set_xticklabels(genes, rotation=rotate_xticks, ha="right")

        # y-limits and left axis ticks (positive only)
        line_pad    = 0.04
        sig_gap     = 0.015
        delta_gap   = 0.065

        needed_tops = [max(float(r["p_ecDNA"]), float(r["p_chr7"])) + line_pad + sig_gap + delta_gap
                       for _, r in dfm.iterrows()]
        upper = max(1.05, max(needed_tops, default=1.05) + 0.02)
        lower = -1.05 if plot_exon else 0.0
        ax.set_ylim(lower, upper)

        pos_ticks = np.linspace(0, 1, 6)
        ax.set_yticks(pos_ticks)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Bursting (%)")

        # right axis only when plotting exon bars
        if plot_exon:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            tick_vals = np.linspace(lo, hi, 5)
            tick_pos  = -((tick_vals - lo) / rng)
            ax2.set_yticks(tick_pos)
            ax2.set_yticklabels([f"{v:.2f}" for v in tick_vals])
            ax2.set_ylabel("Pseudobulk exon expression (log10 CPM)")

        # annotate % inside bars
        def _annotate_inside(bars, vals):
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    y = max(v - 0.02, 0.02)
                    ax.text(b.get_x() + b.get_width()/2, y, f"{v:.0%}",
                            ha="center", va="top", fontsize=8)
        _annotate_inside(bars_ec,   dfm["p_ecDNA"].to_numpy(float))
        _annotate_inside(bars_chr7, dfm["p_chr7"].to_numpy(float))

        # significance + delta annotations between bars
        for i, r in dfm.iterrows():
            h_left  = float(r["p_ecDNA"])
            h_right = float(r["p_chr7"])
            y_line  = max(h_left, h_right) + line_pad

            ax.plot([x[i] - width/2, x[i] + width/2], [y_line, y_line], color="k", lw=0.8)
            sig = _p_to_stars(r["pval"])
            ax.text(x[i], y_line + sig_gap, sig, ha="center", va="bottom", fontsize=10)

            d = float(r["p_ecDNA"] - r["p_chr7"])
            ax.text(x[i], y_line + sig_gap + delta_gap, f"Δ={d:+.0%}",
                    ha="center", va="bottom", fontsize=9, color="black")

        ax.legend(loc="upper right", frameon=False)

        if title is None:
            title = ("Bursting percentage (top) with per-gene Fisher significance and Δ(ecDNA−chr7)"
                     + ("" if not plot_exon else "\nand pseudobulk exon expression (bottom)"))
        ax.set_title(title)

        plt.tight_layout()

        # --- save as SVG if requested ---
        if save_svg_path is not None:
            fig.savefig(save_svg_path, format="svg", bbox_inches="tight")
            print(f"Saved SVG to: {save_svg_path}")

        return fig, ax

    finally:
        # restore rcParams so we don't globally change your session
        plt.rcParams.update(old_rc)

import statsmodels.api as sm

def plot_genes_vs_rgs_percent_bins_binnedR2(
    df_circular,
    x_col='total_genes_transcribed',
    y_col='rgs',
    n_groups=20,
    tick_positions=[25, 50, 75, 100],
    line_color='#4169E1',     # hex color for fitted line
    save_svg=False,
    svg_path=None,
    title=None,               # optional title
    ylim=None,                # option to set y limits
    figsize=(4, 3),
    dpi=150,
    n_boot=2000,              # bootstrap samples for CI of median
    ci=95,                    # CI level for median
    seed=None,                # optional random seed
):
    # --- preserve and set rcParams (Arial + text in SVG) ---
    old_rc = plt.rcParams.copy()
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'   # keep text as <text> in SVG

    rng = np.random.default_rng(seed)

    df = df_circular[[x_col, y_col]].dropna().copy()

    # Normalize to %
    xmax = df[x_col].max()
    df['pct'] = df[x_col] / xmax * 100.0

    # Make equal-width bins (0–100)
    edges = np.linspace(0, 100, n_groups + 1)
    df['bin'] = pd.cut(df['pct'], bins=edges, include_lowest=True, labels=False)

    # --- compute median + 95% CI of median per bin (bootstrap) ---
    grouped = df.groupby('bin')[y_col]

    rows = []
    alpha = 100 - ci
    for b, vals in grouped:
        arr = vals.to_numpy()
        n = len(arr)
        if n == 0:
            continue
        med = np.median(arr)

        if n > 1:
            boot = rng.choice(arr, size=(n_boot, n), replace=True)
            boot_meds = np.median(boot, axis=1)
            lo, hi = np.percentile(boot_meds, [alpha/2, 100 - alpha/2])
        else:
            lo = hi = med

        rows.append({
            'bin': b,
            'median': med,
            'ci_lo': lo,
            'ci_hi': hi,
            'count': n,
        })

    summary = pd.DataFrame(rows).set_index('bin').sort_index()

    bins = summary.index.values.astype(int)
    x_centers = (edges[:-1] + edges[1:]) / 2
    x_centers = x_centers[bins]

    y_med   = summary['median'].values
    yerr_lo = y_med - summary['ci_lo'].values
    yerr_hi = summary['ci_hi'].values - y_med

    # --- regression on binned medians ---
    X_bin = x_centers
    Y_bin = y_med
    Xmat  = sm.add_constant(X_bin)
    model = sm.OLS(Y_bin, Xmat).fit()

    slope, intercept = model.params[1], model.params[0]
    r2, pval_reg = model.rsquared, model.pvalues[1]

    # --- Pearson correlation on binned medians ---
    if len(X_bin) > 1:
        r_pearson, p_pearson = pearsonr(X_bin, Y_bin)
    else:
        r_pearson, p_pearson = np.nan, np.nan

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # dots + 95% CI whiskers (median at center)
    ax.errorbar(
        X_bin, Y_bin,
        yerr=[yerr_lo, yerr_hi],
        fmt='o', color='black',
        ecolor='gray',
        elinewidth=1.4,
        capsize=3
    )

    # fitted line
    xx = np.linspace(0, 100, 300)
    yy = intercept + slope * xx
    ax.plot(xx, yy, color=line_color, lw=2)

    # stats text (show both R² and Pearson)
    ax.text(
        0.05, 0.95,
        f"y = {slope:.3f}x + {intercept:.2f}\n"
        f"$R^2$ = {r2:.3f}\n"
        f"Pearson r = {r_pearson:.3f}\n"
        f"p (Pearson) = {p_pearson:.2e}",
        transform=ax.transAxes, va='top', fontsize=9
    )

    # axis formatting
    ax.set_xlim(0, 100)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_positions])
    ax.set_xlabel("Genes transcribed (%)")
    ax.set_ylabel("Rg (µm)")

    # apply y-limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)

    # optional title
    if title is not None:
        ax.set_title(title)

    # clean spines
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    plt.tight_layout()

    # optional SVG saving
    if save_svg:
        if svg_path is None:
            raise ValueError("svg_path must be provided when save_svg=True")
        fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # restore rcParams
    plt.rcParams.update(old_rc)

    return fig, ax, summary, model, r_pearson, p_pearson

from matplotlib.colors import Normalize
from scipy.stats import pearsonr
def _bootstrap_ci_median(x, n_boot=2000, ci=95, seed=None):
    """
    Bootstrap CI for the median.
    Returns (median, low_CI, high_CI)
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]

    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    else:
        idx = np.random.randint(0, len(x), size=(n_boot, len(x)))

    samples = x[idx]
    med_boot = np.median(samples, axis=1)

    median_val = np.median(x)
    alpha = (100 - ci) / 2
    low = np.percentile(med_boot, alpha)
    high = np.percentile(med_boot, 100 - alpha)

    return median_val, low, high
def dotplot_bootstrap_median_with_pearson(
    df_plot,
    *,
    x_col,
    y_col,
    z_col,
    group_col,
    n_groups=None,
    y_norm=None,
    z_norm=None,              # NEW: divide z by this
    z_to_percent=False,        # NEW: if True, z becomes 0-100 scale
    n_boot=2000,
    ci=95,
    cmap="rainbow",
    figsize=(5, 4),
    s=90,
    alpha=0.9,
    edgecolor="black",
    linewidth=0.8,
    title=None,
    x_label=None,
    y_label=None,
    z_label=None,
    font_family="Arial",
    save_svg=None,
    dpi=800,
    seed=None,
    ax=None
):
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.stats import pearsonr

    # ---------- SVG-friendly font ----------
    prev_font = mpl.rcParams.get("font.family", None)
    prev_svg  = mpl.rcParams.get("svg.fonttype", None)
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["svg.fonttype"] = "none"

    # ---------- select + deduplicate columns ----------
    needed = list(set([x_col, y_col, z_col, group_col]))
    df = df_plot[needed].dropna()

    # ---------- decide z normalization ----------
    # (compute default z_norm from data if requested)
    z_scale = 1.0
    if z_to_percent:
        if z_norm is None:
            z_norm = float(pd.to_numeric(df[z_col], errors="coerce").max())
        z_scale = 100.0

    if z_norm is not None:
        z_norm = float(z_norm) if z_norm not in [0, None, np.nan] else 1.0

    # ---------- grouping ----------
    if n_groups is None:
        df["_group"] = df[group_col]
        groups = sorted(df["_group"].unique())
        group_ranks = {g: i for i, g in enumerate(groups)}
    else:
        q = pd.to_numeric(df[group_col], errors="coerce")
        mask = q.notna()
        df = df.loc[mask].copy()
        q = q.loc[mask]

        ranks = q.rank(method="first")  # 1..N unique ranks
        bins = pd.qcut(ranks, q=n_groups, duplicates="raise")
        df["_group"] = bins

        groups = list(df["_group"].cat.categories)
        group_ranks = {g: i for i, g in enumerate(groups)}

    # ---------- per-group stats ----------
    rows = []
    for g in groups:
        sub = df[df["_group"] == g]
        if len(sub) == 0:
            continue

        x_med, x_lo, x_hi = _bootstrap_ci_median(sub[x_col], n_boot=n_boot, ci=ci, seed=seed)
        y_med, y_lo, y_hi = _bootstrap_ci_median(sub[y_col], n_boot=n_boot, ci=ci, seed=seed)

        if y_norm is not None:
            y_med /= y_norm
            y_lo  /= y_norm
            y_hi  /= y_norm

        z_med, z_lo, z_hi = _bootstrap_ci_median(sub[z_col], n_boot=n_boot, ci=ci, seed=seed)

        # ---- NEW: normalize z (and optionally convert to %) ----
        if z_norm is not None:
            z_med = (z_med / z_norm) * z_scale
            z_lo  = (z_lo  / z_norm) * z_scale
            z_hi  = (z_hi  / z_norm) * z_scale
        elif z_to_percent:
            # z_to_percent=True always implies a norm; this is a safety fallback
            z_med *= z_scale
            z_lo  *= z_scale
            z_hi  *= z_scale

        rows.append(dict(
            group=g,
            group_rank=group_ranks[g],
            N=len(sub),
            x_med=x_med, x_lo=x_lo, x_hi=x_hi,
            y_med=y_med, y_lo=y_lo, y_hi=y_hi,
            z_med=z_med, z_lo=z_lo, z_hi=z_hi,
        ))

    stats = pd.DataFrame(rows).sort_values("group_rank").reset_index(drop=True)

    # ---------- Pearson on binned medians ----------
    if len(stats) > 1:
        pearson_r, pearson_p = pearsonr(stats["x_med"], stats["y_med"])
    else:
        pearson_r, pearson_p = np.nan, np.nan
    stats["pearson_r"] = pearson_r
    stats["pearson_p"] = pearson_p

    # ---------- plotting ----------
    vmin, vmax = stats["z_med"].min(), stats["z_med"].max()
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig = ax.figure

    ax.errorbar(
        stats["x_med"], stats["y_med"],
        xerr=[stats["x_med"] - stats["x_lo"], stats["x_hi"] - stats["x_med"]],
        yerr=[stats["y_med"] - stats["y_lo"], stats["y_hi"] - stats["y_med"]],
        fmt="none", ecolor="black", elinewidth=1.1, zorder=2,
    )

    sc = ax.scatter(
        stats["x_med"], stats["y_med"],
        c=stats["z_med"], cmap=cmap, norm=norm,
        s=s, alpha=alpha, edgecolors=edgecolor, linewidth=linewidth, zorder=3,
    )

    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)

    if title is not None:
        ax.set_title(title)

    if np.isfinite(pearson_r):
        ax.text(
            0.02, 0.98,
            f"Pearson r = {pearson_r:.3f}\np = {pearson_p:.2e}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=18,
        )

    cb = plt.colorbar(sc, ax=ax)

    if z_label is None:
        if z_to_percent or (z_norm is not None and np.isclose(z_scale, 100.0)):
            z_label = f"{z_col} (%)"
        else:
            z_label = z_col
    cb.set_label(z_label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_svg:
        fig.savefig(save_svg, format="svg", dpi=dpi, bbox_inches="tight")

    # ---------- restore rcParams ----------
    if prev_font is not None:
        mpl.rcParams["font.family"] = prev_font
    if prev_svg is not None:
        mpl.rcParams["svg.fonttype"] = prev_svg

    return fig, ax, stats


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import kruskal, f_oneway, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests


def seaborn_multi_violin_from_df(
    df,
    value_col,
    group_col,
    *,
    order=None,                 # optional explicit ordering
    colors=None,
    title="",
    x_label = None,
    y_label=None,
    show_counts=True,
    font_family="Arial",
    width=0.8,
    figsize=(5.6, 4.4),
    grid_color="0.9",
    rasterize_violins=False,
    save_svg=None,
    dpi=800,

    # stats
    test="kruskal",             # default is reviewer-safe
    alternative="two-sided",
    do_posthoc=False,
    fdr_method="fdr_bh",

    # mean dot
    show_mean_dot=True,
    mean_dot_size=120,
):
    """
    Multi-violin plot pulling groups directly from a dataframe column.
    """

    # ----- SVG-safe fonts -----
    prev_font = mpl.rcParams.get("font.family", None)
    prev_svg  = mpl.rcParams.get("svg.fonttype", None)
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["svg.fonttype"] = "none"

    # ----- clean -----
    df = df[[value_col, group_col]].dropna()

    if order is None:
        # auto-sort numbers, otherwise keep categorical order
        if np.issubdtype(df[group_col].dtype, np.number):
            order = sorted(df[group_col].unique())
        else:
            order = list(pd.unique(df[group_col]))

    groups = [
        df.loc[df[group_col] == g, value_col].values.astype(float)
        for g in order
    ]

    # remove nonfinite
    groups = [g[np.isfinite(g)] for g in groups]

    if any(len(g) == 0 for g in groups):
        raise ValueError("At least one group is empty.")

    # ----- GLOBAL TEST -----
    if test == "kruskal":
        res = kruskal(*groups)
        stat, pval = float(res.statistic), float(res.pvalue)
        stat_name = "H"
        test_name = "Kruskal–Wallis"

    elif test == "anova":
        res = f_oneway(*groups)
        stat, pval = float(res.statistic), float(res.pvalue)
        stat_name = "F"
        test_name = "One-way ANOVA"

    else:
        raise ValueError("test must be 'kruskal' or 'anova'")

    # ----- POSTHOC -----
    posthoc_df = None
    if do_posthoc:
        pairs = []
        pvals = []

        for i in range(len(order)):
            for j in range(i+1, len(order)):

                if test == "kruskal":
                    r = mannwhitneyu(groups[i], groups[j], alternative=alternative)
                    pv = r.pvalue
                    method = "MWU"
                else:
                    r = ttest_ind(groups[i], groups[j], equal_var=False)
                    pv = r.pvalue
                    method = "Welch t"

                pairs.append((order[i], order[j]))
                pvals.append(pv)

        _, qvals, _, _ = multipletests(pvals, method=fdr_method)

        posthoc_df = pd.DataFrame({
            "group1":[p[0] for p in pairs],
            "group2":[p[1] for p in pairs],
            "p":pvals,
            "q":qvals,
            "test":method
        }).sort_values("q")

    # ----- COLORS -----
    if colors is None:
        palette = sns.color_palette(n_colors=len(order))
    else:
        palette = colors

    # ----- PLOT -----
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid_color, lw=1)

    sns.violinplot(
        data=df,
        x=group_col,
        y=value_col,
        order=order,
        palette=palette,
        inner=None,              # ← cleaner modern look
        cut=0,
        scale="width",
        linewidth=1.1,
        width=width/2,
        ax=ax,
        zorder=2.5
    )

    # black edges
    for poly in ax.findobj(plt.Polygon):
        poly.set_edgecolor("black")
        poly.set_linewidth(1.1)

    # median dots
    if show_mean_dot:
        med = df.groupby(group_col)[value_col].mean().reindex(order)

        ax.scatter(
            np.arange(len(order)),
            med.values,
            s=mean_dot_size,
            color="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=3
        )

    # counts
    if show_counts:
        counts = df[group_col].value_counts().reindex(order)
        ax.set_xticklabels([f"{g}\n(n={counts[g]:,})" for g in order])

    # labels
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # expand ylim for stats text
    ymin, ymax = df[value_col].min(), df[value_col].max()
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.05*yr, ymax + 0.20*yr)

    ax.text(
        1.2, ymax + 0.13*yr,
        f"{test_name}: {stat_name}={stat:.2g}, p={pval:.2g}",
        ha="center",
        fontsize=14
    )

    # rasterize violins
    if rasterize_violins:
        from matplotlib.collections import PolyCollection
        for coll in ax.collections:
            if isinstance(coll, PolyCollection):
                coll.set_rasterized(True)

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_svg:
        fig.savefig(save_svg, format="svg", dpi=dpi, bbox_inches="tight")

    # restore rc
    if prev_font is not None:
        mpl.rcParams["font.family"] = prev_font
    if prev_svg is not None:
        mpl.rcParams["svg.fonttype"] = prev_svg

    return {
        "test": test_name,
        "stat": stat,
        "p": pval,
        "posthoc": posthoc_df
    }


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
def plot_group_mean_scatter_with_size(
    df,
    xcol="rgs_trimmed",
    ycol="total_genes_transcribed_3genes",
    zcol="zone_1indexed",
    size_col=None,                      # ← NOW OPTIONAL
    size_stat="mean",
    s_min=100,
    s_max=600,
    cmap_name="viridis_r",
    add_reg_line=True,
    rasterize_scatter=False,
    save_svg=None,
    figsize=(15, 5),

    # labels
    x_label=None,
    y_label=None,
    group_label=None,
    z_label=None,

    # title + stats
    title=None,
    add_pearson=True,
    pearson_loc=(0.02, 0.98),
    pearson_fontsize=None,

    # ticks
    x_major_step=0.05,
    y_major_step=0.2,
    style="white",

    # font behavior
    font_scale=None,
    tick_labelsize=None,
    label_fontsize=None,
    legend_fontsize=None,
    legend_title_fontsize=None,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    if font_scale is not None:
        base = mpl.rcParams
        rc_updates = {
            "font.size": base["font.size"] * font_scale,
            "axes.labelsize": base["axes.labelsize"] * font_scale,
            "axes.titlesize": base["axes.titlesize"] * font_scale,
            "xtick.labelsize": base["xtick.labelsize"] * font_scale,
            "ytick.labelsize": base["ytick.labelsize"] * font_scale,
            "legend.fontsize": base["legend.fontsize"] * font_scale,
            "legend.title_fontsize":
                base.get("legend.title_fontsize", base["legend.fontsize"]) * font_scale,
        }
        rc_ctx = mpl.rc_context(rc_updates)
    else:
        rc_ctx = mpl.rc_context()

    with rc_ctx:
        sns.set(style=style)

        # ======================
        # summarize
        # ======================
        required_cols = [zcol, xcol, ycol]
        if size_col is not None:
            required_cols.append(size_col)

        g = (
            df.dropna(subset=required_cols)
              .groupby(zcol, observed=True)
              .agg(
                  x_mean=(xcol, "mean"),
                  y_mean=(ycol, "mean"),
                  **(
                      {"size_val": (size_col, size_stat)}
                      if size_col is not None else {}
                  )
              )
              .reset_index()
        )

        if g.empty:
            raise ValueError("No rows remain after dropna/groupby.")

        zone_num = pd.to_numeric(g[zcol].astype(str), errors="coerce")
        if zone_num.isna().any():
            zone_num = pd.Categorical(g[zcol]).codes + 1

        g = (
            g.assign(zone_num=zone_num)
             .sort_values("zone_num")
             .reset_index(drop=True)
        )

        # ======================
        # size mapping
        # ======================
        if size_col is None:
            sizes = np.full(len(g), (s_min + s_max) / 2)

            def map_size(val):
                return (s_min + s_max) / 2
        else:
            v = pd.to_numeric(g["size_val"], errors="coerce").to_numpy()
            lo, hi = np.nanpercentile(v, [5, 95])

            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                sizes = np.full_like(v, (s_min + s_max) / 2, dtype=float)

                def map_size(val):
                    return (s_min + s_max) / 2
            else:
                v_clip = np.clip(v, lo, hi)
                sizes = s_min + (v_clip - lo) / (hi - lo) * (s_max - s_min)

                def map_size(val):
                    val = np.clip(val, lo, hi)
                    return s_min + (val - lo) / (hi - lo) * (s_max - s_min)

        # ======================
        # colors
        # ======================
        cmap = plt.get_cmap(cmap_name)
        K = len(g)
        colors = [cmap(i / max(K - 1, 1)) for i in range(K)]

        # ======================
        # plot
        # ======================
        fig, ax = plt.subplots(figsize=figsize)

        if add_reg_line and K >= 3:
            sns.regplot(
                data=g,
                x="x_mean",
                y="y_mean",
                scatter=False,
                ci=95,
                color="#666666",
                line_kws=dict(linewidth=1.5, alpha=0.9),
                ax=ax,
            )
            for ln in ax.lines:
                ln.set_zorder(1)

        ax.scatter(
            g["x_mean"], g["y_mean"],
            s=sizes,
            c=colors,
            edgecolors="white",
            linewidths=0.9,
            alpha=0.75,
            zorder=3,
        )

        ax.set_xlabel(x_label if x_label else f"Mean {xcol}", fontsize=label_fontsize)
        ax.set_ylabel(y_label if y_label else f"Mean {ycol}", fontsize=label_fontsize)

        if title is not None:
            ax.set_title(title)

        # ======================
        # Pearson
        # ======================
        if add_pearson:
            try:
                from scipy.stats import pearsonr
                x = pd.to_numeric(g["x_mean"], errors="coerce").to_numpy()
                y = pd.to_numeric(g["y_mean"], errors="coerce").to_numpy()
                ok = np.isfinite(x) & np.isfinite(y)
                if np.sum(ok) >= 2:
                    r, p = pearsonr(x[ok], y[ok])
                    ptxt = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
                    stat_txt = f"Pearson r = {r:.2f}\np = {ptxt}"
                else:
                    stat_txt = "Pearson r: NA\np: NA\nN < 2"
            except Exception:
                stat_txt = "Pearson r: NA\np: NA"

            ax.text(
                pearson_loc[0], pearson_loc[1],
                stat_txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=pearson_fontsize,
            )

        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(False)

        ax.xaxis.set_major_locator(MultipleLocator(x_major_step))
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        # ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.tick_params(axis="both", which="major",
                       direction="out", length=6, width=1.2,
                       labelsize=tick_labelsize)
        ax.tick_params(axis="both", which="minor",
                       direction="out", length=3, width=0.9)
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            bottom=True,
            top=False,        # ← turn on top ticks
            left=True,
            right=False       # ← turn on right ticks
        )


        # ======================
        # legends
        # ======================
        z_title = z_label if z_label else zcol

        color_handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=colors[i], markeredgecolor="white",
                   markersize=8, label=str(g.loc[i, zcol]))
            for i in range(K)
        ]

        leg1 = ax.legend(
            handles=color_handles,
            title=z_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            frameon=True,
        )
        ax.add_artist(leg1)  # <-- keep color legend no matter what

        if size_col is not None:
            size_title = group_label if group_label else size_col
            qvals = np.nanpercentile(v, [10, 50, 90])
            qvals = np.unique(np.round(qvals, 3))

            size_handles = [
                ax.scatter([], [], s=map_size(q),
                           facecolors="none", edgecolors="k", linewidths=0.8)
                for q in qvals
            ]

            ax.legend(
                size_handles,
                [str(q) for q in qvals],
                title=size_title,
                loc="upper left",
                bbox_to_anchor=(1.02, 0.55),
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                frameon=True,
            )

        plt.tight_layout(rect=[0, 0, 0.78, 1])

        if rasterize_scatter:
            for coll in ax.collections:
                if isinstance(coll, PathCollection):
                    coll.set_rasterized(True)

        if save_svg:
            fig.savefig(save_svg, format="svg", dpi=1200,
                        bbox_inches="tight", transparent=True)

        return fig, ax, g
