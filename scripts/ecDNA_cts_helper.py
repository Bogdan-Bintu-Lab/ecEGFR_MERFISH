#### This script is a collection of helper functions that is used to get the counts per round per cell
#### the core idea is to get the drift-corrected spots, get bright spots, and collapse them in a single matrix
#### the code here has been tested in GBM39/Human_P4CC_EGFR_12_17_2024/060225_P4CCDecCore_median_DNA_cts.ipynb
import os,glob
import numpy as np
from tqdm import tqdm
import sys
from ioMicroS6A import *
def fov_to_ifov(fov):
    return int(fov.split('_')[-1]) ## update 082625
def get_htag(fl): return os.path.basename(fl).split('--')[1].split('_set')[0]
def get_icol(fl): return int(fl.split('--col')[-1].split('_')[0])
def get_hindex(fl): return int(get_htag(fl).split('_')[0][1:])
    
def is_RNA(fl): return ('R' in get_htag(fl)) and (get_hindex(fl) > 1)
def is_DNA(fl): return get_htag(fl).startswith('D')
def get_dic_drifts(fov, 
           drift_fls = None):  
    if drift_fls is None:
        drift_fls = glob.glob(os.path.join(analysis_fld, 'drift*'))
    ## get drift dic
    drift_fl = [fl for fl in drift_fls if fov in fl][0]
    print(drift_fl)
    # Load drift data
    drifts, flds, fov_, fl_ref = np.load(drift_fl, allow_pickle=True)
    dic_drifts = {os.path.basename(fld).split('_set')[0]: drft[0] for fld, drft in zip(flds, drifts)}

    return dic_drifts

def get_dic_segm(segm_fld):
    segm_fls = glob.glob(os.path.join(segm_fld, rf'*.npz'))
    fovs = [os.path.basename(fl).split('--')[0] for fl in segm_fls]
    dic_segm_fls = dict(zip(fovs, segm_fls))
    return dic_segm_fls


def build_counts_matrix(fov, segm, htag, icol, nuc_ids, cell_ids, speckle_ids):
    """
    Builds a counts matrix with rows as unique cells (excluding 0) and columns:
    [ifovcell, {htag}-counts_nuc, {htag}-counts_cell, {htag}-counts_speckle]

    Parameters:
        fov (str or identifier): Field of view name or code (input to fov_to_ifov).
        segm (array-like): Segmentation mask with cell labels.
        htag (str): Header tag to prefix to counts columns.
        nuc_ids (list or array): Cell IDs assigned to nuclei.
        cell_ids (list or array): Cell IDs assigned to general cell objects.
        speckle_ids (list or array): Cell IDs assigned to speckles.

    Returns:
        counts_matrix (np.ndarray): Matrix of shape (N_cells-1, 4).
        header (list): Column names.
    """
    # Get all unique cells and remove 0
    all_cells = np.unique(segm)
    all_cells = all_cells[all_cells != 0]

    # Get ifov from fov
    ifov = fov_to_ifov(fov)

    # Compute ifovcell
    ifovcells = ifov * 10**5 + all_cells

    # Count objects for each type
    nuc_cells, nuc_counts = np.unique(nuc_ids, return_counts=True)
    cell_cells, cell_counts = np.unique(cell_ids, return_counts=True)
    speckle_cells, speckle_counts = np.unique(speckle_ids, return_counts=True)

    # Initialize the output matrix
    counts_matrix = np.zeros((len(all_cells), 4), dtype=int)
    counts_matrix[:, 0] = ifovcells

    # Build lookup dictionaries
    nuc_dict = dict(zip(nuc_cells, nuc_counts))
    cell_dict = dict(zip(cell_cells, cell_counts))
    speckle_dict = dict(zip(speckle_cells, speckle_counts))

    # Populate the count columns
    for i, icell in enumerate(all_cells):
        counts_matrix[i, 1] = nuc_dict.get(icell, 0)
        counts_matrix[i, 2] = cell_dict.get(icell, 0)
        counts_matrix[i, 3] = speckle_dict.get(icell, 0)

    header = [
        'ifovcell',
        f'{htag}_col{icol}-counts_nuc',
        f'{htag}_col{icol}-counts_cell',
        f'{htag}_col{icol}-counts_speckle'
    ]
    
    return counts_matrix, header


def get_XH_DNA_single(fov, htag,set_,
                    icol,
                       dic_drifts,
                dic_segm_fls,
               th_h = np.exp(9), 
               chrom_mat = [0.601659, -0.0017476836, 0.06975247],
              analysis_fld = r'Y:\\RenImaging2\\Brett\\Human_P4CC_EGFR_12_17_2024_EGFR_Analysis',
            save_fld = 'Y:\dragonfruit2\Human_P4CC_EGFR_12_17_2024_median_DNA',
              force = False,
                   suffix = '_redo',
                     im_size = [2900, 2900]):
    '''
    Assuming colocalization of two colors
    '''
    save_fl = os.path.join(f'{save_fld}{suffix}', f'{fov}--{htag}--col{icol}--X_cts{suffix}.npz')
    
    # Check if saved file exists and handle force logic
    if os.path.exists(save_fl) and not force:
        print(f"Loading existing counts matrix from {save_fl}")
        data = np.load(save_fl, allow_pickle=True)
        counts_matrix = data['cts_mat']
        header = data['cts_header'].tolist()
        X = counts_matrix.copy()
        return counts_matrix, header, X
    
    ## else, perform the computation
    ## load raw fits
    fit_fl = os.path.join(analysis_fld, f'{fov}--{htag}_{set_}--col{icol}__Xhfits.npz')
    XH = np.load(fit_fl,allow_pickle = True)['Xh']
    ## this is to get rid of the weird mermake bug, that when fitting, it will include few extra points
    keep_mask = (XH[:, 1] <= im_size[0]) & (XH[:, 2] <= im_size[1])
    XH = XH[keep_mask]
    ## brightness threshold 
    ## speckle threshold
    from scipy.spatial import KDTree
    keep = (XH[:,-1]>th_h)
    
    ## chromatic abberation
    XH_ = XH[keep]
    XH_[:,:3]=XH_[:,:3]-chrom_mat[icol]
    
    ## drift correction to segm
    XH_[:,:3] = XH_[:,:3]+dic_drifts[htag]
    
    ## load segm
    segm_fl = dic_segm_fls[fov]
    segm_keys = list(np.load(segm_fl).keys())
    if 'dapi_segm' in segm_keys:
        segm_str = 'dapi_segm'
    else:
        segm_str = 'segm'
    segm = np.load(segm_fl)[segm_str]
    shape = np.load(segm_fl)['shape']
    resc = segm.shape/shape
    
    ##  rescale XH
    X = XH_[:,:3]
    X_red = np.round(X*resc).astype(int)

    ## assign points to the nucleus
    nuc_ids = np.zeros(len(X_red),dtype=int)
    keep_im = np.all((X_red>=0)&(X_red<segm.shape),axis=-1)
    nuc_ids[keep_im]=segm[tuple(X_red[keep_im].T)]
    
    ## assign points to the cytoplasm
    if 'cyto_segm' in segm_keys:
        segm_ =np.load(segm_fl)['cyto_segm']
    else:
        segm_ = expand_segmentation(segm,10)
    cell_ids = np.zeros(len(X_red),dtype=int)
    keep_im = np.all((X_red>=0)&(X_red<segm.shape),axis=-1)
    cell_ids[keep_im]=segm_[tuple(X_red[keep_im].T)]
    
    ## record potential speckles
    bad = (XH_[:,-4]<0)
    speckle_ids = np.zeros(len(X_red[bad]),dtype=int)
    keep_im = np.all((X_red[bad]>=0)&(X_red[bad]<segm.shape),axis=-1)
    speckle_ids[keep_im]=segm_[tuple(X_red[bad][keep_im].T)]

    ## format everything in a cell x meta matrix
    valid = nuc_ids > 0
    X_ = np.column_stack((X[valid], nuc_ids[valid]))
    counts_matrix, header = build_counts_matrix(fov, segm, htag,icol, nuc_ids, cell_ids, speckle_ids)
 
    ## save result
    np.savez_compressed(save_fl, X = X, XH = XH_,
                        nuc_ids = nuc_ids, cell_ids = cell_ids, speckle_ids = speckle_ids,
                        cts_mat = counts_matrix, cts_header = header)
    print(f'Saved to {save_fl}')
    return counts_matrix, header, X_

from functools import reduce
def assemble_counts_fov_redo(fov, save_fld, tags, save_fls = None, force = False, suffix = '_redo'):
    """
    Assemble counts matrices for a given FOV from .npz files into a single DataFrame.

    Parameters:
        fov (str): FOV identifier to match in filenames.
        save_fld (str): Folder containing the .npz files.

    Returns:
        cts_fov (pd.DataFrame): Concatenated DataFrame with ifovcell as index.
    """
    save_fl = os.path.join(f'{save_fld}{suffix}', f'{fov}--DNA_X_cts_fov{suffix}.csv')

    # Check if saved file exists and handle force logic
    if os.path.exists(save_fl) and not force:
        print(f"Loading existing counts matrix from {save_fl}")
        merged_df = pd.read_csv(save_fl)
        merged_df = merged_df.set_index('fov_ifovcell')
        return merged_df
    
    # Find all matching files
    if save_fls:
        fls_keep = save_fls
    else:
        pattern = os.path.join(save_fld, f"{fov}*X_cts{suffix}.npz")
        fls = glob.glob(pattern)

        ## keep only those matching the tags
        fl_tags = [os.path.basename(fl).split('--')[1] for fl in fls]
        fls_keep = fls[np.isin(fl_tags, tags)]
        print(fls_keep)
    cts_dfs = []
    for fl in fls_keep:
        data = np.load(fl, allow_pickle=True)
        cts_mat = data['cts_mat']
        cts_header = data['cts_header']
        cts_df = pd.DataFrame(cts_mat, columns = cts_header)
        cts_dfs.append(cts_df)

    if not cts_dfs:
        raise ValueError(f"No matching files found for fov {fov} in {save_fld}")

   # Perform a successive merge on 'ifovcell'
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='ifovcell', how='outer'), cts_dfs)
    
    ## add fov to the ifovcell
    merged_df['fov_ifovcell'] = fov + '_' + merged_df['ifovcell'].astype(str)
    
    # Optionally set ifovcell as index
    merged_df = merged_df.set_index('fov_ifovcell')
    
   ## save to file
    merged_df.to_csv(save_fl)
    print(f"saved FOV counts matrix to {save_fl}")
    return merged_df