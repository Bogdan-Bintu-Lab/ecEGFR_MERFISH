"""
Neftel-style scoring + butterfly pipeline (clean, annotated)
===========================================================

This module provides a tidy, reproducible implementation of the workflow you
iterated to:

- Signature filtering against `adata.var` (optionally drop zero/constant genes)
- Neftel/Tirosh-style scoring with expression-matched controls by bins
  (random sampling or deterministic union of bin-mates)
- AnnData wrappers (correct orientation: cells×genes in AnnData → genes×cells
  for the scorer)
- Butterfly coordinates with your baseline mapping:
      NPC = bottom-left, OPC = bottom-right, AC = top-left, MES = top-right
- State assignment from butterfly X/Y
- Bootstrapping utilities:
  • group-level composition CIs
  • cell-wise consensus (mean X/Y, per-state probabilities, entropy)
- Plotting helpers (white bg, equal axes, right-side legend)

All functions are typed and documented. You can import the public API from the
bottom of this file.

Author: you + ChatGPT
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global palette & constants
# -----------------------------------------------------------------------------

NEFTEL_STATES = ("AC", "MES", "OPC", "NPC", "Ambiguous")
PALETTE = {
    "AC": "#54A24B",       # green
    "MES": "#F58518",      # orange
    "OPC": "#7E57C2",      # purple
    "NPC": "#4C78A8",      # blue
    "Ambiguous": "#A0A0A0", # gray
}

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int | None) -> None:
    """Set numpy's RNG seed (no-op if None)."""
    if seed is not None:
        np.random.seed(int(seed))


def _row_means(mat: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Row means for dense/sparse matrices (genes × cells)."""
    return np.asarray(mat.mean(axis=1)).ravel() if sp.issparse(mat) else mat.mean(axis=1)


def _col_means(mat: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Column means for dense/sparse matrices (genes × cells → per-cell mean)."""
    return np.asarray(mat.mean(axis=0)).ravel() if sp.issparse(mat) else mat.mean(axis=0)


# -----------------------------------------------------------------------------
# Signature filtering
# -----------------------------------------------------------------------------

def filter_signatures_to_var(
    adata,
    signatures: Mapping[str, Sequence[str]],
    *,
    layer: str | None = None,
    use_raw: bool = False,
    drop_zero_total: bool = True,
    drop_zero_var: bool = True,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Intersect signatures with genes present in `adata` and optionally drop
    genes with zero total or zero variance in the chosen matrix.

    Parameters
    ----------
    adata : AnnData
    signatures : dict name -> list of genes
    layer : use this layer for filtering (cells×genes)
    use_raw : if True and `adata.raw` exists, use `adata.raw.X`
    drop_zero_total : drop genes with total sum == 0 across cells
    drop_zero_var : drop genes with variance == 0 across cells
    verbose : print per-signature keep/missing stats
    """
    if use_raw and (adata.raw is not None):
        X = adata.raw.X
        genes = pd.Index(adata.raw.var_names.astype(str))
    else:
        X = adata.layers[layer] if layer is not None else adata.X
        genes = pd.Index(adata.var_names.astype(str))

    keep = pd.Series(True, index=genes)
    if drop_zero_total:
        totals = np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else X.sum(axis=0)
        keep &= totals > 0
    if drop_zero_var:
        if sp.issparse(X):
            V = np.array([np.var(np.asarray(X[:, i].todense()).ravel()) for i in range(X.shape[1])])
        else:
            V = X.var(axis=0)
        keep &= V > 0

    valid = set(genes[keep])
    out: Dict[str, List[str]] = {}
    for name, gset in signatures.items():
        gset = [str(g) for g in gset]
        kept = sorted(valid.intersection(gset))
        out[name] = kept
        if verbose:
            missing = sorted(set(gset) - set(genes))
            filtered = sorted(set(gset) - set(kept) - set(missing))
            print(f"[{name}] kept {len(kept)} (missing: {len(missing)}, zero/const filtered: {len(filtered)})")
    return out


# -----------------------------------------------------------------------------
# Binning & control construction
# -----------------------------------------------------------------------------

def bin_genes(
    *,
    mat: np.ndarray | sp.spmatrix | None = None,
    x: Optional[pd.Series] = None,
    gene_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> pd.Series:
    """Create ~equal-sized bins across genes by mean expression.

    Provide either `x` (Series of per-gene values, index=genes) or a matrix
    `mat` (genes×cells) with `gene_names`. Returns a pd.Series(index=gene,
    values=int bin_id).
    """
    if x is None:
        if mat is None or gene_names is None:
            raise ValueError("Provide `x` or (`mat` and `gene_names`).")
        x = pd.Series(_row_means(mat), index=pd.Index(gene_names, name="gene"))
    else:
        if x.index is None:
            if gene_names is None:
                raise ValueError("`x` must be indexed by gene, or pass `gene_names`.")
            x.index = pd.Index(gene_names, name="gene")

    x_sorted = x.sort_values(kind="mergesort")
    chunks = np.array_split(x_sorted.index.values, n_bins)
    return pd.Series({g: b for b, idxs in enumerate(chunks) for g in idxs}, name="bin_id", dtype=int)


def bin_genes_by_detection(
    mat: np.ndarray | sp.spmatrix,
    gene_names: Sequence[str],
    n_bins: int = 10,
) -> pd.Series:
    """Alternative binning: by detection rate (fraction of cells > 0)."""
    if sp.issparse(mat):
        det = np.asarray((mat > 0).mean(axis=1)).ravel()
    else:
        det = (mat > 0).mean(axis=1)
    x = pd.Series(det, index=pd.Index(gene_names, name="gene"))
    x_sorted = x.sort_values(kind="mergesort")
    chunks = np.array_split(x_sorted.index.values, n_bins)
    return pd.Series({g: b for b, idxs in enumerate(chunks) for g in idxs}, dtype=int)


def binmatch(
    group: Sequence[str],
    *,
    bins: pd.Series,
    n: int = 15,
    replace: bool = True,
) -> List[str]:
    """For each gene in `group`, sample `n` control genes from the same bin."""
    group = list(group)
    missing = pd.Index(group).difference(bins.index)
    if len(missing):
        raise KeyError(f"Missing group genes in bins: {list(missing)}")

    by_bin: Dict[int, List[str]] = {}
    for g, b in bins.items():
        by_bin.setdefault(int(b), []).append(g)

    controls: List[str] = []
    for g in group:
        b = int(bins.loc[g])
        candidates = by_bin[b]
        pool = [c for c in candidates if c != g] if not replace else candidates
        if not replace and len(pool) < n:
            controls.extend(list(np.random.choice(candidates, size=n, replace=True)))
        else:
            controls.extend(list(np.random.choice(pool, size=n, replace=replace)))
    return controls


def deterministic_controls_from_bins(
    groups: Mapping[str, Sequence[str]],
    bins: pd.Series,
) -> Dict[str, List[str]]:
    """Deterministic control sets: union of all bin-mates of signature genes
    (excluding the signature genes themselves)."""
    by_bin: Dict[int, List[str]] = {}
    for g, b in bins.items():
        by_bin.setdefault(int(b), []).append(g)

    ctrl: Dict[str, List[str]] = {}
    for name, genes in groups.items():
        sig = [g for g in genes if g in bins.index]
        mates = [set(by_bin[int(bins[g])]) for g in sig]
        union = set().union(*mates) if mates else set()
        ctrl[name] = sorted(list(union.difference(sig)))
    return ctrl


# -----------------------------------------------------------------------------
# Scoring (genes × cells)
# -----------------------------------------------------------------------------

def _mean_for_set(mat, gene_names: Sequence[str], subset: Sequence[str]) -> np.ndarray:
    idx = pd.Index(gene_names).get_indexer(list(subset))
    if (idx < 0).any():
        missing = [list(subset)[i] for i, j in enumerate(idx) if j < 0]
        raise KeyError(f"Missing genes in matrix: {missing}")
    sub = mat[idx, :]  # works for dense & sparse
    return _col_means(sub)


def score_core(
    mat: np.ndarray | sp.spmatrix,
    gene_names: Sequence[str],
    groups: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    controls: Mapping[str, Sequence[str]] | Sequence[str] | bool | None = None,
    center: bool = False,
    center_by: str = "mean",
) -> pd.DataFrame:
    """Score = mean(group) − mean(controls).

    `controls` can be:
      - None/False → no subtraction
      - True       → subtract global mean across all genes
      - dict/list  → subtract mean of provided control gene set(s)
    Returns a (cells × groups) DataFrame.
    """
    if isinstance(groups, Mapping):
        grp = {k: list(v) for k, v in groups.items()}
    else:
        grp = {"score": list(groups)}

    mode = "none"
    ctrl_dict: Optional[Dict[str, List[str]]] = None
    if controls is True:
        mode = "global"
        global_ctrl = _col_means(mat)
    elif isinstance(controls, Mapping):
        mode = "dict"
        ctrl_dict = {k: list(v) for k, v in controls.items()}
        if set(ctrl_dict) != set(grp):
            raise ValueError("controls keys must match groups keys")
    elif isinstance(controls, Sequence) and not isinstance(controls, (str, bytes)):
        if len(grp) != 1:
            raise ValueError("List `controls` only allowed for single-group scoring.")
        mode = "dict"
        ctrl_dict = {next(iter(grp)): list(controls)}

    out: Dict[str, np.ndarray] = {}
    for name, genes in grp.items():
        present = [g for g in genes if g in set(gene_names)]
        if not present:
            out[name] = np.zeros(mat.shape[1], dtype=float)
            continue
        sig = _mean_for_set(mat, gene_names, present)
        if mode == "none":
            score = sig
        elif mode == "global":
            score = sig - global_ctrl
        else:
            cgenes = [g for g in ctrl_dict[name] if g in set(gene_names)]
            ctrl = _mean_for_set(mat, gene_names, cgenes) if cgenes else 0.0
            score = sig - ctrl
        out[name] = score

    scores = pd.DataFrame(out)
    if center:
        if center_by == "mean":
            scores -= scores.mean(axis=0)
        elif center_by == "median":
            scores -= scores.median(axis=0)
        else:
            by = np.asarray(center_by, float)
            if by.shape[0] != scores.shape[1]:
                raise ValueError("center_by length mismatch")
            scores -= by
    return scores


def score(
    mat: np.ndarray | sp.spmatrix,
    gene_names: Sequence[str],
    groups: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    binmat: np.ndarray | sp.spmatrix | None = None,
    bins: Optional[pd.Series] = None,
    controls: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    bin_control: bool = False,
    center: bool = False,
    n_bins: int = 10,
    n: int = 15,
    replace: bool = True,
) -> pd.DataFrame:
    """Wrapper: optionally auto-build bin-matched controls and score."""
    if (bins is not None or binmat is not None) and controls is None:
        bin_control = True
    if controls is not None:
        bin_control = False

    if isinstance(groups, Mapping):
        grp = {k: list(v) for k, v in groups.items()}
    else:
        grp = {"score": list(groups)}

    ctrl_dict = None
    if bin_control:
        if bins is None:
            bins = bin_genes(mat=binmat if binmat is not None else mat, gene_names=gene_names, n_bins=n_bins)
        ctrl_dict = {name: binmatch([g for g in genes if g in bins.index], bins=bins, n=n, replace=replace)
                     for name, genes in grp.items()}
    else:
        ctrl_dict = controls

    return score_core(mat=mat, gene_names=gene_names, groups=grp, controls=ctrl_dict, center=center)


# -----------------------------------------------------------------------------
# AnnData wrappers (cells × genes → transpose)
# -----------------------------------------------------------------------------

def score_anndata(
    adata,
    groups: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    layer: str | None = None,
    use_raw: bool | None = None,
    gene_pool: Sequence[str] | None = None,
    bin_from_layer: str | None = None,
    n_bins: int = 10,
    n: int = 15,
    replace: bool = True,
    center: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Score many signatures on an AnnData; writes to `.obs` if `prefix` is set."""
    if use_raw is None:
        use_raw = adata.raw is not None
    if use_raw:
        X = adata.raw.X  # cells × genes
        var_names = adata.raw.var_names
    else:
        X = adata.layers[layer] if layer is not None else adata.X
        var_names = adata.var_names

    if bin_from_layer is not None:
        BIN = adata.layers[bin_from_layer]  # cells × genes
        bin_names = adata.var_names
    else:
        BIN = X
        bin_names = var_names

    if gene_pool is not None:
        pool = pd.Index(gene_pool, dtype=str)
        keep_sc = pd.Index(var_names).intersection(pool)
        keep_bn = pd.Index(bin_names).intersection(pool)
    else:
        keep_sc = pd.Index(var_names)
        keep_bn = pd.Index(bin_names)

    sc_order = pd.Index(var_names).get_indexer(keep_sc)
    bn_order = pd.Index(bin_names).get_indexer(keep_bn)

    X_use = X[:, sc_order]          # cells × genes
    BIN_use = BIN[:, bn_order]      # cells × genes

    gnames_sc = keep_sc.astype(str).tolist()
    gnames_bn = keep_bn.astype(str).tolist()

    # Build bins on genes × cells
    bins = bin_genes(mat=BIN_use.T, gene_names=gnames_bn, n_bins=n_bins)

    # Score on genes × cells
    scores = score(
        mat=X_use.T,
        gene_names=gnames_sc,
        groups=groups,
        bins=bins,
        binmat=None,
        controls=None,
        bin_control=True,
        center=center,
        n_bins=n_bins,
        n=n,
        replace=replace,
    )  # cells × groups

    if scores.shape[0] != adata.n_obs:
        raise ValueError(f"Scores rows ({scores.shape[0]}) != adata cells ({adata.n_obs})")

    if prefix is not None:
        for col in scores.columns:
            adata.obs[f"{prefix}{col}"] = pd.Series(scores[col].values, index=adata.obs_names)
    return scores


def score_anndata_deterministic(
    adata,
    groups: Mapping[str, Sequence[str]],
    *,
    layer: str | None = None,
    use_raw: bool | None = None,
    bin_from_layer: str | None = None,
    n_bins: int = 10,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Like `score_anndata`, but uses full-bin deterministic controls (no sampling)."""
    if use_raw is None:
        use_raw = adata.raw is not None
    if use_raw:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.layers[layer] if layer is not None else adata.X
        var_names = adata.var_names

    if bin_from_layer is not None:
        BIN = adata.layers[bin_from_layer]
        bin_names = adata.var_names
    else:
        BIN = X
        bin_names = var_names

    bins = bin_genes(mat=BIN.T, gene_names=list(map(str, bin_names)), n_bins=n_bins)
    controls = deterministic_controls_from_bins(groups, bins)

    scores = score_core(
        mat=X.T,
        gene_names=list(map(str, var_names)),
        groups=groups,
        controls=controls,
        center=False,
    )

    if prefix is not None:
        for col in scores.columns:
            adata.obs[f"{prefix}{col}"] = pd.Series(scores[col].values, index=adata.obs_names)
    return scores


# -----------------------------------------------------------------------------
# Butterfly (your baseline mapping) & state assignment
# -----------------------------------------------------------------------------

def compute_butterfly(
    scores: pd.DataFrame,
    *,
    ac: str = "AC",
    mes: str = "MES",
    opc: str = "OPC",
    npc: str = "NPC",
    log_scale: bool = True,
) -> pd.DataFrame:
    """Compute butterfly coordinates using baseline mapping:
    bl = NPC, br = OPC, tl = AC, tr = MES.
    Returns DataFrame with columns X, Y (index = cells).
    """
    req = [ac, mes, opc, npc]
    missing = [c for c in req if c not in scores.columns]
    if missing:
        raise KeyError(f"Missing score columns: {missing}")

    bl, br, tl, tr = scores[npc].values, scores[opc].values, scores[ac].values, scores[mes].values
    bottom = np.maximum(bl, br)
    top = np.maximum(tl, tr)
    x = np.where(bottom > top, br - bl, tr - tl)
    y = top - bottom

    if log_scale:
        X = np.sign(x) * np.log2(np.abs(x) + 1.0)
        Y = np.sign(y) * np.log2(np.abs(y) + 1.0)
    else:
        X, Y = x, y
    return pd.DataFrame({"X": X, "Y": Y}, index=scores.index)


def assign_neftel_state(
    adata,
    *,
    x_col: str = "butterfly_X",
    y_col: str = "butterfly_Y",
    out_col: str = "Neftel_state",
    x_eps: float = 0.0,
    y_eps: float = 0.0,
    ambig_label: str = "Ambiguous",
) -> pd.Categorical:
    """Assign states from butterfly X/Y using baseline quadrant mapping.

    TL: AC, TR: MES, BR: OPC, BL: NPC. Points within |X|≤x_eps or |Y|≤y_eps
    are labeled Ambiguous.
    """
    x = pd.to_numeric(adata.obs[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(adata.obs[y_col], errors="coerce").to_numpy()

    lab = np.full(x.shape, ambig_label, dtype=object)
    top = y > y_eps
    bot = y < -y_eps
    lab[top & (x < -x_eps)] = "AC"
    lab[top & (x > x_eps)] = "MES"
    lab[bot & (x > x_eps)] = "OPC"
    lab[bot & (x < -x_eps)] = "NPC"

    adata.obs[out_col] = pd.Categorical(lab, categories=list(NEFTEL_STATES))
    return adata.obs[out_col]


# -----------------------------------------------------------------------------
# Bootstrapping (group-level & cellwise)
# -----------------------------------------------------------------------------

def proportions_by_group(adata, group_col: str = "leiden", state_col: str = "Neftel_state") -> pd.DataFrame:
    """Row-normalized % composition of `state_col` within each `group_col`."""
    ct = pd.crosstab(adata.obs[group_col], adata.obs[state_col]).sort_index()
    return (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0)


def bootstrap_neftel_proportions(
    adata,
    signatures_f: Mapping[str, Sequence[str]],
    *,
    group_col: str = "leiden",
    n_boot: int = 50,
    n_bins: int = 10,
    n: int = 15,
    replace: bool = True,
    seeds: Optional[Sequence[int]] = None,
    prefix: str = "Neftel_",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap the random controls to estimate mean/SD/CI of stacked % bars."""
    if seeds is None:
        seeds = list(range(n_boot))

    tables: List[pd.DataFrame] = []
    for b, seed in enumerate(seeds[:n_boot]):
        set_seed(seed)
        scores = score_anndata(
            adata,
            groups=signatures_f,
            n_bins=n_bins,
            n=n,
            replace=replace,
            center=False,
            prefix=None,
        )
        df = scores.rename(columns={"AC": "AC", "MES": "MES", "OPC": "OPC", "NPC": "NPC"})
        H = compute_butterfly(df, log_scale=True)
        tmpX, tmpY, tmpState = f"__bx_{b}", f"__by_{b}", f"__st_{b}"
        adata.obs[tmpX] = H["X"].values
        adata.obs[tmpY] = H["Y"].values
        assign_neftel_state(adata, x_col=tmpX, y_col=tmpY, out_col=tmpState)
        tbl = proportions_by_group(adata, group_col=group_col, state_col=tmpState)
        tbl["__boot__"] = b
        tables.append(tbl)
        del adata.obs[tmpX], adata.obs[tmpY], adata.obs[tmpState]

    big = pd.concat(tables, axis=0).reset_index().rename(columns={"index": group_col})
    value_cols = [c for c in big.columns if c not in [group_col, "__boot__"]]
    long = big.melt(id_vars=[group_col, "__boot__"], value_vars=value_cols, var_name="state", value_name="pct")
    summary = (
        long.groupby([group_col, "state"])['pct']
            .agg(mean='mean', sd='std')
            .reset_index()
    )
    summary["ci95_lo"] = summary["mean"] - 1.96 * (summary["sd"] / np.sqrt(n_boot))
    summary["ci95_hi"] = summary["mean"] + 1.96 * (summary["sd"] / np.sqrt(n_boot))
    return summary.pivot(index=group_col, columns="state"), long


def bootstrap_cellwise_neftel(
    adata,
    signatures_f: Mapping[str, Sequence[str]],
    *,
    n_boot: int = 50,
    n_bins: int = 10,
    n: int = 15,
    replace: bool = True,
    seeds: Optional[Sequence[int]] = None,
    ac: str = "AC",
    mes: str = "MES",
    opc: str = "OPC",
    npc: str = "NPC",
    x_eps: float = 0.0,
    y_eps: float = 0.0,
    write_to_adata: bool = True,
    prefix: str = "boot_",
) -> Dict[str, pd.Series | pd.DataFrame]:
    """Per-cell bootstrap aggregation: mean/sd of X/Y, state probabilities,
    consensus state/confidence, and entropy. Writes to AnnData by default."""
    states_order = [ac, mes, opc, npc, "Ambiguous"]
    state_to_idx = {s: i for i, s in enumerate(states_order)}
    n_cells = adata.n_obs

    sumX = np.zeros(n_cells); sumY = np.zeros(n_cells)
    sumX2 = np.zeros(n_cells); sumY2 = np.zeros(n_cells)
    counts = np.zeros((n_cells, len(states_order)), dtype=int)

    if seeds is None:
        seeds = list(range(n_boot))

    for b, seed in enumerate(seeds[:n_boot]):
        set_seed(seed)
        scores = score_anndata(adata, groups=signatures_f, n_bins=n_bins, n=n, replace=replace, center=False, prefix=None)
        df = scores.rename(columns={ac: ac, mes: mes, opc: opc, npc: npc})
        H = compute_butterfly(df, ac=ac, mes=mes, opc=opc, npc=npc, log_scale=True)
        X = H["X"].to_numpy(); Y = H["Y"].to_numpy()

        lab = np.full(n_cells, "Ambiguous", dtype=object)
        top = Y > y_eps; bot = Y < -y_eps
        lab[top & (X <  -x_eps)] = ac
        lab[top & (X >   x_eps)] = mes
        lab[bot & (X >   x_eps)] = opc
        lab[bot & (X <  -x_eps)] = npc

        sumX += X; sumY += Y
        sumX2 += X * X; sumY2 += Y * Y
        idxs = np.fromiter((state_to_idx.get(s, state_to_idx["Ambiguous"]) for s in lab), int, count=n_cells)
        counts[np.arange(n_cells), idxs] += 1

    meanX = sumX / n_boot; meanY = sumY / n_boot
    sdX = np.sqrt(np.maximum(sumX2 / n_boot - meanX**2, 0.0))
    sdY = np.sqrt(np.maximum(sumY2 / n_boot - meanY**2, 0.0))

    probs = counts / n_boot
    max_idx = probs.argmax(axis=1)
    max_prob = probs[np.arange(n_cells), max_idx]
    consensus = np.array(states_order, dtype=object)[max_idx]

    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)

    prob_cols = [f"{prefix}P_{s}" for s in states_order]
    df_probs = pd.DataFrame(probs, columns=prob_cols, index=adata.obs_names)
    ser_consensus = pd.Series(consensus, index=adata.obs_names, name=f"{prefix}consensus_state")
    ser_conf = pd.Series(max_prob, index=adata.obs_names, name=f"{prefix}consensus_conf")
    ser_ent = pd.Series(entropy, index=adata.obs_names, name=f"{prefix}state_entropy")
    df_xy = pd.DataFrame({f"{prefix}X_mean": meanX, f"{prefix}Y_mean": meanY, f"{prefix}X_sd": sdX, f"{prefix}Y_sd": sdY}, index=adata.obs_names)

    if write_to_adata:
        adata.obs[ser_consensus.name] = pd.Categorical(ser_consensus)
        adata.obs[ser_conf.name] = ser_conf
        adata.obs[ser_ent.name] = ser_ent
        for c in df_probs.columns:
            adata.obs[c] = df_probs[c].values
        adata.obsm[f"{prefix}butterfly_mean"] = df_xy[[f"{prefix}X_mean", f"{prefix}Y_mean"]].to_numpy()
        adata.obsm[f"{prefix}butterfly_sd"] = df_xy[[f"{prefix}X_sd", f"{prefix}Y_sd"]].to_numpy()

    return {
        "probs": df_probs,
        "xy_stats": df_xy,
        "consensus_state": ser_consensus,
        "consensus_conf": ser_conf,
        "state_entropy": ser_ent,
    }

import numpy as np
import pandas as pd

def mark_ambiguous_by_conf(
    adata,
    *,
    conf_col: str = "boot_consensus_conf",
    base_state_col: str = "boot_consensus_state",
    flag_col: str = "boot_is_ambiguous",
    out_state_col: str | None = "boot_consensus_state_thr",
    threshold: float = 0.7,
    ambiguous_label: str = "Ambiguous",
    categories: list[str] | None = None,   # optional custom order
):
    """
    Mark cells with consensus confidence < threshold as ambiguous.
    - writes a boolean flag in `obs[flag_col]`
    - optionally writes relabeled states in `obs[out_state_col]`
    """
    if conf_col not in adata.obs:
        raise KeyError(f"Missing column: obs['{conf_col}']")
    if base_state_col not in adata.obs:
        raise KeyError(f"Missing column: obs['{base_state_col}']")

    conf = pd.to_numeric(adata.obs[conf_col], errors="coerce")
    flag = conf < float(threshold)
    adata.obs[flag_col] = flag

    if out_state_col is not None:
        base = adata.obs[base_state_col].astype(str).to_numpy()
        new = base.copy()
        new[flag.values] = ambiguous_label
        if categories is None:
            categories = ["AC", "MES", "OPC", "NPC", ambiguous_label]
        adata.obs[out_state_col] = pd.Categorical(new, categories=categories)

    # remember the cutoff you used
    adata.uns[f"{flag_col}_threshold"] = float(threshold)
    return flag


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_butterfly_scatter(
    adata,
    *,
    x_col: str = "butterfly_X",
    y_col: str = "butterfly_Y",
    state_col: str = "Neftel_state",
    s: int | float = 6,
    alpha: float = 0.85,
    snap: float = 1.0,
    figsize: Tuple[float, float] = (6, 6),
    legend_right: bool = True,
):
    """Scatter butterfly X/Y colored by `state_col` with equal axes and right legend."""
    x = pd.to_numeric(adata.obs[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(adata.obs[y_col], errors="coerce").to_numpy()
    state = adata.obs[state_col].astype(str).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & pd.notna(state)
    x, y, state = x[ok], y[ok], state[ok]

    m = float(np.nanmax([np.abs(x).max(initial=0), np.abs(y).max(initial=0), 1e-9]))
    lim = snap * np.ceil((m * 1.05) / snap)

    order = list(NEFTEL_STATES)
    fig, ax = plt.subplots(figsize=figsize)
    for lab in order:
        sel = (state == lab)
        if sel.any():
            ax.scatter(x[sel], y[sel], s=s, alpha=alpha, c=PALETTE.get(lab, "#000000"), label=lab)

    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.axhline(0, lw=1, c="black", alpha=0.4); ax.axvline(0, lw=1, c="black", alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Butterfly X"); ax.set_ylabel("Butterfly Y")

    if legend_right:
        ax.legend(title=state_col, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True)
        plt.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        ax.legend(title=state_col, frameon=True, fancybox=True); plt.tight_layout()
    return ax


def plot_bootstrap_consensus_scatter(
    adata,
    *,
    xy_key: str = "boot_butterfly_mean",
    state_col: str = "boot_consensus_state",
    s: int | float = 8,
    alpha: float = 0.9,
    snap: float = 1.0,
    figsize: Tuple[float, float] = (6, 6),
    legend_right: bool = True,
):
    """Scatter of mean bootstrapped butterfly coordinates colored by consensus state."""
    XY = np.asarray(adata.obsm[xy_key])
    x = XY[:, 0]; y = XY[:, 1]
    state = adata.obs[state_col].astype(str).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & pd.notna(state)
    x, y, state = x[ok], y[ok], state[ok]

    m = float(np.nanmax([np.abs(x).max(initial=0), np.abs(y).max(initial=0), 1e-9]))
    lim = snap * np.ceil((m * 1.05) / snap)

    order = list(NEFTEL_STATES)
    fig, ax = plt.subplots(figsize=figsize)
    for lab in order:
        sel = (state == lab)
        if sel.any():
            ax.scatter(x[sel], y[sel], s=s, alpha=alpha, c=PALETTE.get(lab, "#000000"), label=lab)

    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.axhline(0, lw=1, c="black", alpha=0.4); ax.axvline(0, lw=1, c="black", alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Butterfly X (boot mean)"); ax.set_ylabel("Butterfly Y (boot mean)")

    if legend_right:
        ax.legend(title=state_col, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True)
        plt.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        ax.legend(title=state_col, frameon=True, fancybox=True); plt.tight_layout()
    return ax


def plot_spatial_by_state(
    adata,
    *,
    label_col: str = "Neftel_state",
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    invert_matplotlib_y: bool = False,
    point_size_other: int = 8,
    point_size_focus: int = 28,
):
    """Facet spatial scatter by state (one figure per state)."""
    xy = np.asarray(adata.obsm["X_spatial"])  # cells × 2
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_y:
        y = -y

    lab = adata.obs[label_col].astype(str)
    states_pref = list(NEFTEL_STATES)
    states = [s for s in states_pref if s in lab.unique()] + [s for s in sorted(lab.unique()) if s not in states_pref]

    for state in states:
        m_focus = lab.values == state
        m_other = ~m_focus
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
        ax.scatter(x[m_other], y[m_other], c="lightgray", s=point_size_other, marker=".", linewidth=0, alpha=0.35)
        ax.scatter(x[m_focus], y[m_focus], c=[PALETTE.get(state, "#000000")], s=point_size_focus, marker=".", linewidth=0, alpha=0.95)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Spatial: {label_col} = {state}", fontsize=16, color="white")
        ax.scatter([], [], c="lightgray", s=point_size_focus, marker=".", label="Other")
        ax.scatter([], [], c=[PALETTE.get(state, "#000000")], s=point_size_focus, marker=".", label=state)
        ax.legend(frameon=False, loc="upper right")
        if invert_matplotlib_y:
            ax.invert_yaxis()
        plt.tight_layout(); plt.show()


def stacked_percentage_bar(
    adata,
    x_col: str,
    y_col: str,
    *,
    x_order: Optional[List[str]] = None,
    y_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (9, 5),
    legend_right: bool = True,
    show_labels: bool = True,
    label_fmt: str = "{p:.0f}%",
    min_pct_label: float = 5.0,
    sort_x_by: str | None = None,  # None | "name" | "count"
):
    """Stacked percentage bars of `y_col` composition within each `x_col` group."""
    obs = adata.obs[[x_col, y_col]].copy()
    x = obs[x_col]; y = obs[y_col]

    if x_order is None:
        x_order = list(x.astype("category").cat.categories) if pd.api.types.is_categorical_dtype(x) else sorted(map(str, x.astype(str).unique()))
    if y_order is None:
        y_order = list(y.astype("category").cat.categories) if pd.api.types.is_categorical_dtype(y) else sorted(map(str, y.astype(str).unique()))

    if sort_x_by == "name":
        try:
            x_order = sorted(x_order, key=lambda s: (str(s).isdigit(), int(s) if str(s).isdigit() else str(s)))
        except Exception:
            x_order = sorted(map(str, x_order))
    elif sort_x_by == "count":
        counts = obs.groupby(x_col).size().reindex(x_order).fillna(0)
        x_order = counts.sort_values(ascending=False).index.tolist()

    df = pd.crosstab(obs[x_col], obs[y_col]).reindex(index=x_order, columns=y_order).fillna(0)
    df_pct = (df.div(df.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0)

    if palette is None:
        palette = {k: PALETTE.get(k, plt.cm.tab10(i % 10)) for i, k in enumerate(y_order)}
    else:
        for i, k in enumerate(y_order):
            palette.setdefault(k, plt.cm.tab10(i % 10))

    fig, ax = plt.subplots(figsize=figsize)
    bottoms = np.zeros(len(x_order), dtype=float)
    xs = np.arange(len(x_order))

    for cat in y_order:
        heights = df_pct[cat].values
        ax.bar(xs, heights, bottom=bottoms, color=palette[cat], edgecolor="white", linewidth=0.5, label=str(cat))
        if show_labels:
            mids = bottoms + heights / 2.0
            for i, (h, m) in enumerate(zip(heights, mids)):
                if h >= min_pct_label:
                    ax.text(xs[i], m, label_fmt.format(p=h), ha="center", va="center", fontsize=8, color="black")
        bottoms += heights

    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.grid(False)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in x_order], rotation=0)
    ax.set_ylabel(f"{y_col} (%)")
    ax.set_xlabel(x_col)
    ax.set_ylim(0, 100)

    if legend_right:
        ax.legend(title=y_col, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True)
        plt.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        ax.legend(title=y_col, frameon=True, fancybox=True)
        plt.tight_layout()

    return ax, df_pct


def refine_consensus_by_center_box(
    adata,
    *,
    xy_key: str = "boot_butterfly_mean",
    base_col: str = "boot_consensus_state",
    out_col: str = "boot_consensus_state_refined",
    halfwidth: float = 0.5,
    halfheight: float | None = None,
    ambig_label: str = "Ambiguous",
    categories: Sequence[str] = NEFTEL_STATES,
):
    """Re-label any cell with |X| ≤ halfwidth and |Y| ≤ halfheight as Ambiguous."""
    if halfheight is None:
        halfheight = halfwidth
    XY = np.asarray(adata.obsm[xy_key])
    X, Y = XY[:, 0], XY[:, 1]

    base = adata.obs[base_col].astype(str).to_numpy()
    refined = base.copy()
    mask = (np.abs(X) <= halfwidth) & (np.abs(Y) <= halfheight)
    refined[mask] = ambig_label
    adata.obs[out_col] = pd.Categorical(refined, categories=list(categories))
    return adata.obs[out_col]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # filtering
    "filter_signatures_to_var",
    # binning & controls
    "bin_genes",
    "bin_genes_by_detection",
    "binmatch",
    "deterministic_controls_from_bins",
    # scoring
    "score_core",
    "score",
    "score_anndata",
    "score_anndata_deterministic",
    # butterfly & states
    "compute_butterfly",
    "assign_neftel_state",
    # bootstrapping
    "proportions_by_group",
    "bootstrap_neftel_proportions",
    "bootstrap_cellwise_neftel",
    "mark_ambiguous_by_conf",
    # plotting
    "plot_butterfly_scatter",
    "plot_bootstrap_consensus_scatter",
    "plot_spatial_by_state",
    "stacked_percentage_bar",
    "refine_consensus_by_center_box",
    # utils
    "NEFTEL_STATES",
    "PALETTE",
    "set_seed",
]
