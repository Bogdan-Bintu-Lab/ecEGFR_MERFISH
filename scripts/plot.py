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

