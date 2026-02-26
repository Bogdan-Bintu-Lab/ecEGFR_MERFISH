import os
import numpy as np
import pandas as pd

def get_dic_drifts(fov, 
           drift_fls = None):  
    if drift_fls is None:
        drift_fls = glob.glob(os.path.join(analysis_fld, 'drift*'))
    ## get drift dic
    drift_fl = [fl for fl in drift_fls if fov in fl][0]
    # print(drift_fl)
    # Load drift data
    drifts, flds, fov_, fl_ref = np.load(drift_fl, allow_pickle=True)
    dic_drifts = {os.path.basename(fld).split('_set')[0]: drft[0] for fld, drft in zip(flds, drifts)}

    return dic_drifts


def flat_field_image_zscore(
    fov,
    iA,
    fld,
    flat_field_tag,
    use_robust_std=False,
    precomputed_scale=None,
    deconv = False
):
    """
    Flat-field correct and normalize an image.

    Parameters
    ----------
    fov : str
        Filename of the raw image (whatever read_im expects).
    iA : int
        Channel index (1-based) used to pick the color: (iA-1) % 3.
    fld : str
        Folder containing the raw image.
    flat_field_tag : str
        Prefix/path for flat-field files, e.g. 'C:/.../flatfield_'.
        Will load: flat_field_tag + 'med_col_raw{icol}.npz'
    use_robust_std : bool, optional
        If True, use MAD-based robust std instead of np.std.
    precomputed_scale : float or None
        If provided, skip computing std/MAD and just divide by this.

    Returns
    -------
    im_norm : np.ndarray, float32
        Flat-field corrected and normalized image, same shape as im__.
    """
    icol = (iA - 1) % 3

    # --- load & flat-field correct ---
    im_ = read_im(os.path.join(fld, fov))      # shape: (C, H, W) or similar
    im__ = np.array(im_[icol], dtype=np.float32)  

    fl_med = f"{flat_field_tag}med_col_raw{icol}.npz"
    im_med = np.array(np.load(fl_med)["im"], dtype=np.float32)
    im_med = cv2.blur(im_med, (20, 20)) # (20, 20)

    # flat-field correction: bring illumination to median of im_med
    im__ = im__ / im_med * np.median(im_med)
    #### deconvolve here
    if deconv:
        im__ = full_deconv(im__,
                            s_=500,
                            pad=100,
                            psf=np.load(r'C:\WholeGenome\NMERFISH_Jenny\psfs\psf_750_Scope4_final.npy'),
                            parameters={'method': 'wiener', 'beta': 0.0025, 'niter': 50},
                            gpu=True,
                            force=True,
                        )
    
    
    im_norm = np.array([im__T-cv2.blur(im__T,(20,20)) for im__T in im__]) # DEBUG: (20,20)
    im_norm = (im_norm - np.median(im_norm)) / np.std(im_norm)
    return im_norm.astype(np.float32)

def extract_trace_intensity_from_3d_image_manual(
    traces,
    im_norm,
    pix_size,
    drift_vec,
):
    """
    Extract per-locus intensities from a 3D image for each trace.

    Parameters
    ----------
    traces : np.ndarray, shape (N, 252, 6)
        Trace coordinates; first 3 columns are (z, x, y) in microns.
    im_norm : np.ndarray, shape (Z, Y, X)
        3D normalized image (e.g. (30, 2800, 2800)).
    pix_size : float or sequence of 3 floats
        Pixel size in microns. If scalar, assumed isotropic for z,x,y.
        Order is (z_pix_size, x_pix_size, y_pix_size) to match (z,x,y).
    drift_vec : array-like of length 3
        Drift in pixel units (z, x, y) to subtract AFTER converting to pixels.

    Returns
    -------
    intensities : np.ndarray, shape (N, 252), dtype float32
        Per-locus image intensity. NaN where input locus was NaN or out-of-bounds.
    """
    traces = np.asarray(traces, dtype=np.float32)
    im_norm = np.asarray(im_norm, dtype=np.float32)

    N, L, D = traces.shape
    assert L == 252, "Expected 252 loci in traces"
    assert D >= 3, "traces must have at least 3 coords (z, x, y)"

    Zdim, H, W = im_norm.shape  # (z, y, x)

    # --- pixel size vector (for z,x,y) ---
    pix_vec = np.array(pix_size, dtype=np.float32)
    if pix_vec.size == 1:
        pix_vec = np.repeat(pix_vec, 3)
    elif pix_vec.size != 3:
        raise ValueError("pix_size must be a scalar or a sequence of length 3 (z,x,y)")

    # --- drift vector (z,x,y) in pixel units ---
    drift_vec = np.array(drift_vec, dtype=np.float32)
    if drift_vec.size != 3:
        raise ValueError("drift_vec must have length 3 (z,x,y) in pixels")

    # --- extract z,x,y (µm) and convert to pixel space ---
    zxy = traces[:, :, :3]  # shape: (N, 252, 3) as (z,x,y)
    valid = ~np.isnan(zxy).any(axis=-1)  # (N, 252): True when all 3 coords are finite

    # microns -> pixels
    zxy_pix = zxy / pix_vec  # broadcast (N,252,3) / (3,)
    # drift correction (subtract drift in pixels)
    zxy_pix_corr = zxy_pix - drift_vec  # broadcast (N,252,3) - (3,)

    # split into components
    z_pix = zxy_pix_corr[:, :, 0]
    x_pix = zxy_pix_corr[:, :, 1]
    y_pix = zxy_pix_corr[:, :, 2]

    # round to nearest integer indices
    z_idx = np.rint(z_pix).astype(np.int64)
    x_idx = np.rint(x_pix).astype(np.int64)
    y_idx = np.rint(y_pix).astype(np.int64)

    # in-bounds mask for 3D image
    in_bounds = (
        (z_idx >= 0) & (z_idx < Zdim) &
        (y_idx >= 0) & (y_idx < H) &
        (x_idx >= 0) & (x_idx < W)
    )

    # final mask: valid coords AND inside image
    sample_mask = valid & in_bounds

    # --- allocate output and fill with NaNs ---
    intensities = np.full((N, L), np.nan, dtype=np.float32)

    # flatten masks and indices for vectorized sampling
    flat_mask = sample_mask.ravel()
    flat_z = z_idx.ravel()[flat_mask]
    flat_y = y_idx.ravel()[flat_mask]
    flat_x = x_idx.ravel()[flat_mask]

    # sample from im_norm[z, y, x]
    sampled_vals = im_norm[flat_z, flat_x, flat_y]

    # put back into (N, L)
    intensities.ravel()[flat_mask] = sampled_vals.astype(np.float32)

    return intensities

import os
import numpy as np
import pandas as pd

def compute_normalized_brightness_manual(
    df_meta,
    pool_to_traces,
    dic_im_norm,
    dic_drifts_ifov,
    pix_size,
    ifov_base=10**5,
    col_pool='pool',
    col_global_row='global_row',
    name='normalized_brightness',
    # --- NEW ---
    save_dir=None,          # e.g. r"D:\..."; if None, no saving
    iA=None,                # used in filename; required if save_dir is not None
    overwrite=True,        # skip existing unless True
    save_dtype=np.float32,  # dtype to save
):
    """
    For each row in df_meta, compute NaN-median brightness across loci of its trace.
    Optionally save per-trace per-locus intensity vectors as .npy files.

    Saved filenames:
      f"ifovcell{cell_id}_{pool}_globalrow{global_row}_iA{iA}.npy"
    """
    if save_dir is not None:
        if iA is None:
            raise ValueError("If save_dir is provided, you must also provide iA for filename.")
        os.makedirs(save_dir, exist_ok=True)

    s_out = pd.Series(index=df_meta.index, dtype=np.float32)

    for pool_name, traces in pool_to_traces.items():
        mask_pool = df_meta[col_pool] == pool_name
        if not mask_pool.any():
            continue

        df_pool = df_meta[mask_pool]
        global_rows = df_pool[col_global_row].to_numpy().astype(int)

        traces_pool = traces[global_rows]  # (M,252,6)

        # cell_id / ifovcell per trace (from last column)
        ifovcell = np.nanmean(traces_pool[:, :, -1], axis=1)     # (M,)
        ifov = (ifovcell // ifov_base).astype(int)               # (M,)

        for this_ifov in np.unique(ifov):
            if this_ifov not in dic_im_norm:
                continue
            if this_ifov not in dic_drifts_ifov:
                continue

            im_norm = dic_im_norm[this_ifov]       # (Z,X,Y) in your convention
            drift_vec = dic_drifts_ifov[this_ifov]

            fov_mask = (ifov == this_ifov)
            if not np.any(fov_mask):
                continue

            idx_traces = np.where(fov_mask)[0]     # positions into traces_pool / global_rows / ifovcell
            df_idx = df_pool.index[idx_traces]

            traces_sel = traces_pool[idx_traces]               # (K,252,6)
            ifovcell_sel = ifovcell[idx_traces]                # (K,)
            global_rows_sel = global_rows[idx_traces]          # (K,)

            intensities = extract_trace_intensity_from_3d_image_manual(
                traces_sel,
                im_norm=im_norm,
                pix_size=pix_size,
                drift_vec=drift_vec,
            )  # (K,252)

            with np.errstate(all='ignore'):
                mean_brightness = np.nanmean(intensities, axis=1).astype(np.float32)

            s_out.loc[df_idx] = mean_brightness

            # --- NEW: save each row of intensities as its own .npy ---
            if save_dir is not None:
                # Make sure we save in the same order as idx_traces
                for k in range(intensities.shape[0]):
                    cell_id = int(np.round(ifovcell_sel[k]))  # ifovcell should be integer-like
                    g_row = int(global_rows_sel[k])

                    fn = f"ifovcell_{cell_id}-trtype_{pool_name}-globalrow_{g_row}-iA_{iA}.npy"
                    fp = os.path.join(save_dir, fn)

                    if (not overwrite) and os.path.exists(fp):
                        continue

                    np.save(fp, intensities[k].astype(save_dtype, copy=False))

    s_out.name = name
    return s_out

import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_antibody_norm_brightness_zscore_streaming(
    iA,
    df_meta,
    ufovs,
    *,
    # --- core behavior ---
    col_ifov="ifov",
    use_robust_std=False,
    batch_size_fovs=1,

    # --- experiment-specific knobs (replace hardcoding) ---
    fov_from_ifov=None,              # callable: (ifov:int)->str
    drift_globber=None,              # callable: (iA:int)->list[str] OR list[str]
    fld_from_iA=None,                # callable: (iA:int)->str
    flat_field_tag=None,             # str
    dic_iA_flds=None,                # only needed if fld_from_iA uses it

    # --- dependencies / constants used inside compute ---
    pool_to_traces=None,             # dict like {"linear":..., "circular":...}
    pix_size=None,
    ifov_base=10**5,
    col_pool="pool",
    col_global_row="global_row",

    # --- extra passthrough to compute fn (e.g., save_dir, iA) ---
    compute_kwargs=None,
):
    """
    Memory-safe streaming wrapper:
    - builds im_norm + drifts per FOV batch
    - computes brightness only for rows belonging to those FOVs
    - writes results back aligned to df_meta.index
    """

    if fov_from_ifov is None:
        # sensible default: your original zscan1 naming
        fov_from_ifov = lambda ifov: f"Conv_zscan1__{str(int(ifov)).zfill(3)}"

    if drift_globber is None:
        raise ValueError("Provide drift_globber (callable(iA)->list[str] OR list[str]).")

    if fld_from_iA is None:
        raise ValueError("Provide fld_from_iA (callable(iA)->str).")

    if flat_field_tag is None:
        raise ValueError("Provide flat_field_tag (str).")

    if pool_to_traces is None:
        raise ValueError("Provide pool_to_traces, e.g. {'linear': linear_traces, 'circular': circ_traces}.")

    if pix_size is None:
        raise ValueError("Provide pix_size.")

    compute_kwargs = {} if compute_kwargs is None else dict(compute_kwargs)

    # preallocate output (aligns with df_meta index)
    s_out = pd.Series(np.nan, index=df_meta.index, dtype=float)

    # drift files (either list or callable)
    drift_fls = drift_globber(iA) if callable(drift_globber) else list(drift_globber)

    ufovs = list(ufovs)
    for i in tqdm(range(0, len(ufovs), batch_size_fovs)):
        fov_batch = ufovs[i:i + batch_size_fovs]

        dic_im_norm = {}
        dic_drifts_ifov = {}

        # build only for this batch
        for ifov in fov_batch:
            fov = fov_from_ifov(ifov)

            dic_im_norm[ifov] = flat_field_image_zscore(
                fov=fov,
                iA=iA,
                fld=fld_from_iA(iA),
                flat_field_tag=flat_field_tag + os.sep,
                use_robust_std=use_robust_std,
            )

            dic_drifts = get_dic_drifts(fov, drift_fls=drift_fls)
            dic_drifts_ifov[ifov] = dic_drifts[dic_iA_Around[iA]]

        # subset df_meta to only rows in these FOVs
        m = df_meta[col_ifov].isin(fov_batch)
        df_sub = df_meta.loc[m].copy()

        if df_sub.empty:
            dic_im_norm.clear()
            dic_drifts_ifov.clear()
            continue

        # compute only for this subset
        s_sub = compute_normalized_brightness_manual(
            df_meta=df_sub,
            pool_to_traces=pool_to_traces,
            dic_im_norm=dic_im_norm,
            dic_drifts_ifov=dic_drifts_ifov,
            pix_size=pix_size,
            ifov_base=ifov_base,
            col_pool=col_pool,
            col_global_row=col_global_row,
            name="normalized_brightness",
            **compute_kwargs,   # e.g. save_dir=..., iA=iA, ...
        )

        s_out.loc[df_sub.index] = s_sub

        dic_im_norm.clear()
        dic_drifts_ifov.clear()
        del df_sub, s_sub

    out_col = f"iA{iA}_ab_norm_robust_H" if use_robust_std else f"iA{iA}_ab_norm_H"
    df_meta[out_col] = s_out.values
    return df_meta