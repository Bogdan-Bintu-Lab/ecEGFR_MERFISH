from ioMicroS6A import *
###### ---------------------------
###### select FOVs
###### ---------------------------
def extract_bright_spots(fov, data_fld = data_fld, icol = 1, dapi = False):
    im = read_im(data_fld+os.sep+fov)
    im_raw = np.array(im[icol],dtype=np.float32)
    im_n = norm_slice(im_raw,s=30)
    imDn_ = im_n/np.std(im_n)
    Xh = get_local_maxfast_tensor(imDn_,im_raw=im_raw,gpu=True,th_fit=20,delta=2,delta_fit=1)
    
    if dapi:
        im_dapi = np.array(im[3],dtype=np.float32)
        return Xh, imDn_, im_raw, im_dapi
    else:
        return Xh, imDn_, im_raw


import numpy as np
from sklearn.cluster import DBSCAN

def cluster_points_dbscan(Xh, eps=50, min_samples=5, centroid=False):
    """
    Applies DBSCAN clustering on (z, x, y) coordinates from Xh.

    Parameters:
    - Xh (numpy.ndarray): Array containing detected points (at least first three columns must be z, x, y).
    - eps (float): Maximum radius to consider points in the same cluster.
    - min_samples (int): Minimum number of points to form a cluster.
    - centroid (bool): If True, also return cluster centroids.

    Returns:
    - Xh_with_labels (numpy.ndarray): Xh with an additional column for cluster labels.
    - num_clusters (int): Number of clusters found (excluding noise).
    - centroids (numpy.ndarray, optional): (num_clusters, 4) array with cluster_id, z_centroid, x_centroid, y_centroid.
    """
    if Xh.shape[1] < 3:
        raise ValueError("Xh must have at least 3 columns (z, x, y) for clustering.")

    # Extract (z, x, y) coordinates
    coordinates = Xh[:, :3]

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(coordinates)

    # Add cluster labels to Xh
    Xh_with_labels = np.hstack([Xh, labels.reshape(-1, 1)])

    # Count clusters (excluding noise)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label (-1)
    num_clusters = len(unique_labels)

    # Compute centroids if requested
    centroids = None
    if centroid:
        centroids_list = []
        for cluster_id in unique_labels:
            cluster_points = coordinates[labels == cluster_id]
            centroid_z, centroid_x, centroid_y = np.mean(cluster_points, axis=0)
            centroids_list.append([centroid_z, centroid_x, centroid_y, cluster_id])
        
        centroids = np.array(centroids_list)  # Convert to numpy array

    if centroid:
        return Xh_with_labels, num_clusters, centroids
    else:
        return Xh_with_labels, num_clusters

###### ---------------------------
###### select bright spots (DNA)
###### ---------------------------
import os
import glob
import numpy as np

## helper function
def get_htag(fl): return os.path.basename(fl).split('--')[1]
##### this func needs to be updated based on naming convension
def is_chr(fl): return ('R' in get_htag(fl)) and ('_' in get_htag(fl)) # check if chromatin data of type H*R, different than RNA data which is H*Q* -exons or H*I* - introns
def get_hindex(fl): return int(get_htag(fl).replace('_','').split('R')[0][1:])
def get_icol(fl): return int(fl.split('--col')[-1].split('_')[0])
def get_R(fl):
    htag = get_htag(fl)
    Rs = np.array(htag.split('R')[-1].split('_set')[0].split('_'),dtype=int)
    icol = get_icol(fl)
    return Rs[icol]
def apply_colorcor(x,m=None):
    """This applies chromatic abberation correction to order 2
    x is a Nx3 vector of positions (typically 750(-->647))
    m is a matrix computed by function calc_color_matrix
    y is the corrected vector in another channel"""
    if m is None:
        return x
    exps = []
    order_max=10
    for p in range(order_max+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    #find the order
    mx,my = m.shape
    order = int((my-1)/mx)
    assert(my<len(exps))
    x_ = np.array(x)
    # construct A matrix
    exps = exps[:my]
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    diff = [np.dot(A,m_) for m_ in m]
    return x_+np.array(diff).T

def get_XH(dic_fits_chr,dic_drifts,th_h=0,th_cor=0.4, chrom_fl=r'C:\WholeGenome\NMERFISH_Jenny\dic_chromatic_abberation_Jenny_Scope4_4_17_2024.pkl'):
    XH = []
    for R in tqdm(list(dic_fits_chr.keys())):
        save_fl = dic_fits_chr[R]
        icol = get_icol(save_fl)
        Xh = np.load(save_fl,allow_pickle=True)['Xh']
        if len(Xh.shape):
            Xh = Xh[Xh[:,-1]>th_h]
            Xh = Xh[Xh[:,-2]>th_cor]
            if len(Xh):
                ## drift correction
                if dic_drifts is not None:
                    tzxy = dic_drifts[get_htag(save_fl)]
                    Xh[:,:3]+=tzxy# drift correction
                
                ## chromatic abberation
                if chrom_fl is not None:
                    dic_chrom = pickle.load(open(chrom_fl,'rb'))
                    m = dic_chrom.get('m'+str(icol),None)
                    hfactor = dic_chrom.get('hfactor'+str(icol),1)
                    x3d = Xh[:,:3]
                    x3dT = apply_colorcor(x3d,m=m)
                    Xh[:,:3] = x3dT
                    Xh[:,-1] = Xh[:,-1]*hfactor
                
                icolR = np.array([[icol,R]]*len(Xh))
                XH_ = np.concatenate([Xh,icolR],axis=-1)
                XH.extend(XH_)
    XH = np.array(XH)
    return XH

def get_Xh_fov(fov,  
               # set_ = None,
               segm_fld = r'C:\GBM39\TK_FFBBL1_SampleA_1_27_2023__JennySegmentation\merged_segmentation', 
                analysis_folder=r'C:\GBM39\TK_FFBBL1_SampleA_1_27_2023__JennyAnalysis',
                save_fld_traces=r'C:\GBM39\ecDNATracer\fulldata', 
                   force = False, smFISH = False,
              pix_size = [0.3, 0.108333, 0.108333]):
    """
    Processes a Field of View (FOV) for drift correction, chromosome alignment, and cell segmentation.
    
    Parameters:
        ifov (int): Index of the FOV to process.
        analysis_folder (str, optional): Path to the analysis folder. Defaults to provided path.
        save_fld_traces (str, optional): Path to save the processed results. Defaults to provided path.
        smFISH (boolean): allow for singel round processing
    """

    # Load FOVs
    # if set_ is None:
    #     fovs = np.load(os.path.join(analysis_folder, 'fovs__.npy'))
    #     fov = fovs[ifov].replace('.zarr', '')
    # else:
    #     fovs = np.load(os.path.join(analysis_folder, f'fovs___set{set_}.npy'))
    #     fov = fovs[ifov].replace('.zarr', '')

    print(f'Processing FOV = {fov}...')
    
    save_fl_traces = os.path.join(save_fld_traces, f"{fov}--XF.npz")
    if os.path.exists(save_fl_traces):
        if not force:
            print(f'{save_fl_traces} exists already...')
            return
    
    # get all segm files
    segm_fls = glob.glob(os.path.join(segm_fld, '*segm.npz'))
    dic_segm_fls = dict(zip([os.path.basename(fl).split('--')[0] for fl in segm_fls], segm_fls))
    segm_merged_fl = dic_segm_fls[fov]

    # Find files
    fits_fls = glob.glob(os.path.join(analysis_folder, '*__Xhfits.npz'))
    drift_fls = glob.glob(os.path.join(analysis_folder, 'drift*'))
    # print(len(fits_fls))
    ### hardcoded fix
    fits_fls = [fl.replace('H1_P1F123', 'H1_R1_2_3') for fl in fits_fls]

    # Identify drift file for this FOV
    drift_fl = [fl for fl in drift_fls if fov in fl]
    if not drift_fl:
        if not smFISH:
            print(f"No drift file found for FOV: {fov}")
            return
        else: ##smFISH
            dic_drifts = None
    else:
        drift_fl = drift_fl[0]
        # Load drift data
        drifts, flds, fov_, fl_ref = np.load(drift_fl, allow_pickle=True)
        dic_drifts = {os.path.basename(fld): drft[0] for fld, drft in zip(flds, drifts)}

    # Filter fits files
    fits_fov = [fl for fl in fits_fls if fov in fl]
    fits_chr = [fl for fl in fits_fov if is_chr(fl)]  # is_chr needs to be defined
    # print(fits_fov)
    # print(fits_chr)
    
    # Sort and create dictionary
    ordered_fits_chr = np.array(fits_chr)[np.argsort([get_hindex(fl) for fl in fits_chr])]
    dic_fits_chr = {get_R(fl): fl for fl in ordered_fits_chr}  # get_R and get_hindex need to be defined
    ## hardcoded fix
    dic_fits_chr = dict(zip(dic_fits_chr.keys(), [fl.replace('H1_R1_2_3', 'H1_P1F123') for fl in dic_fits_chr.values()]))
    Rs = np.sort(list(dic_fits_chr.keys()))

    
    # Load segmentation files
    if 'cyto_segm' in np.load(segm_merged_fl).keys():
        im_segm = np.load(segm_merged_fl)['cyto_segm']
        im_segmNuc = np.load(segm_merged_fl)['dapi_segm']
        shape = np.load(segm_merged_fl)['shape']
        resc = np.round(shape / im_segm.shape).astype(int)
        im_segm_ = im_segm  # Optionally expand segmentation
    else: ## this is with dapi segm only
        im_segm = np.load(segm_merged_fl)['segm']
        im_segmNuc = np.load(segm_merged_fl)['segm']
        shape = np.load(segm_merged_fl)['shape']
        resc = np.round(shape / im_segm.shape).astype(int)
        im_segm_ = im_segm  # Optionally expand segmentation

    # Get drift-corrected matrix
    XH = get_XH(dic_fits_chr, dic_drifts, th_h=0, th_cor=0.25)  # get_XH needs to be defined
    # print(dic_fits_chr)
    # print(XH.shape)

    # Assign to cells
    icells = np.zeros(len(XH))
    Xred = np.round(XH[:, :3] / resc).astype(int)
    keep = np.all((Xred < im_segm_.shape) & (Xred >= 0), axis=-1)
    icells[keep] = im_segm_[tuple(Xred[keep].T)]

    # Identify nuclear markers
    isnuc = np.zeros(len(XH))
    isnuc[keep] = im_segmNuc[tuple(Xred[keep].T)] > 0

    # Concatenate data
    XF = np.concatenate([XH[icells > 0], isnuc[icells > 0][:, np.newaxis], icells[icells > 0][:, np.newaxis]], axis=-1)
    
    ## swap the order such that
    ## -5: isNuc, -4: H, -3: color, -2: R, -1: cell
    num_cols = XF.shape[1]
    old_idx = num_cols - 2  # -2nd column
    new_idx = num_cols - 5  # -5th position
    # Get column indices
    col_indices = np.arange(num_cols)
    # Remove the -2nd column
    col_indices = np.delete(col_indices, old_idx)
    # Insert the column at the -5th position
    col_indices = np.insert(col_indices, new_idx, old_idx)
    XF_ = XF[:, col_indices]
    
    
    # Save processed data
    os.makedirs(save_fld_traces, exist_ok=True)
    
    np.savez(save_fl_traces, XF=XF_, pix_size=pix_size)

    print(f"Saved processed data for FOV {fov} at {save_fl_traces}")
    return XF_


def get_maxH(XF):
    cells = XF[:,-1].astype(int)
    ucells = np.unique(cells)
    Rs = XF[:,-2].astype(int)
    uRs = np.unique(Rs)
    cols = XF[:,-3].astype(int)
    ucols = np.unique(cols)
    maxH = []
    for uR in tqdm(uRs):
        isR = (Rs==uR)
        XF_ = XF[isR]
        maxH_ = []
        for ucell in ucells:
            iscell = cells[isR]==ucell
            if np.sum(iscell):
                h_ = XF_[iscell][:,-4]
                h__ = np.sort(h_)
                max_ = np.median(h__[-5:])
                min_ = np.median(h__[:5])
                std_ = np.std(h__[:])
                maxH_.append([max_,min_,std_])
        maxH.append(maxH_)
    
    maxH = np.array(maxH)
    medh = np.median(maxH,axis=1)
    return medh,uRs

def get_maxH_nan(XF):
    cells = XF[:,-1].astype(int)
    ucells = np.unique(cells)
    Rs = XF[:,-2].astype(int)
    uRs = np.unique(Rs)
    cols = XF[:,-3].astype(int)
    ucols = np.unique(cols)
    maxH = []
    for uR in tqdm(uRs):
        isR = (Rs == uR)
        XF_ = XF[isR]
        # Initialize maxH_ with default values for each cell
        maxH_ = np.full((len(ucells), 3), np.nan)  # Using NaN to signify no data
        for idx, ucell in enumerate(ucells):
            iscell = cells[isR] == ucell
            if np.sum(iscell):
                h_ = XF_[iscell][:, -4]
                h__ = np.sort(h_)
                max_ = np.median(h__[-5:]) if len(h__) >= 5 else np.nan  # Check sufficient data
                min_ = np.median(h__[:5]) if len(h__) >= 5 else np.nan  # Check sufficient data
                std_ = np.std(h__) if len(h__) > 0 else np.nan  # Check for any data
                maxH_[idx] = [max_, min_, std_]
        maxH.append(maxH_)
    # return maxH
    maxH = np.array(maxH)
    medh = np.nanmedian(maxH,axis=1)
    return medh, uRs
    

## within XF, order has been swapped!!!
def select_britgher_spots(XF,medh,uRs,th_h=5):
    ths = medh[:,1]+th_h*medh[:,2]
    ths_ = np.zeros(np.max(uRs)+1)
    ths_[uRs]=ths
    H = XF[:,-4]
    Rs= XF[:,-2].astype(int)
    keeph = H>ths_[Rs]
    XF_ = XF[keeph]
    return XF_

import numpy as np
from scipy.spatial import KDTree

def find_colocalized_points(XF, dth):
    # Extract zxy coordinates and additional columns
    data_columns = XF[:, :3]  # zxy coordinates

    # Unique R values sorted
    unique_r = np.sort(np.unique(XF[:, -3]))  # Extract from R column directly
    
    # Dictionary to hold the submatrices by R
    submatrices = {r: XF[XF[:, -3] == r] for r in unique_r}
    
    # Dictionary to hold colocalized points
    colocalized_points = {}
    
    # Iterate over each unique R value
    for r in unique_r:
        current_full_matrix = submatrices[r]
        current_matrix = current_full_matrix[:, :3]  # Only zxy coordinates for KDTree
        neighbors = []
        
        # Check for previous neighbor
        if r - 1 in submatrices:
            neighbors.append(submatrices[r - 1][:, :3])  # Only zxy coordinates for KDTree
        
        # Check for next neighbor
        if r + 1 in submatrices:
            neighbors.append(submatrices[r + 1][:, :3])  # Only zxy coordinates for KDTree
        
        # Merge neighbor matrices
        if neighbors:
            neighbor_matrix = np.vstack(neighbors)
            neighbor_tree = KDTree(neighbor_matrix)
            current_tree = KDTree(current_matrix)

            # Find points in neighbor matrix that are within dth of any point in the current matrix
            indices_from_neighbors = neighbor_tree.query_ball_tree(current_tree, dth)

            # Extract unique indices of points in neighbor matrix that are colocalized
            unique_indices = set(idx for sublist in indices_from_neighbors for idx in sublist)

            # Collect colocalized points from neighbor matrix with additional data
            colocalized_points_r = current_full_matrix[list(unique_indices), :] if unique_indices else np.array([])

            colocalized_points[r] = colocalized_points_r
        else:
            # No neighbors, empty result for this R
            colocalized_points[r] = np.array([])
    
    return colocalized_points

def get_rgs(zxys_cast_,pix_size = [0.5,0.108333,0.108333]):
    zxys_cast_ = np.array(zxys_cast_)*pix_size
    zxys_cast_ -= np.nanmean(zxys_cast_,axis=0)[np.newaxis,:]#center of mass
    zxys_cast_ = np.linalg.norm(zxys_cast_,axis=-1)
    rg_cats = np.sqrt(np.nanmean(zxys_cast_**2))
    return rg_cats

def get_dispersion_fov(XF_col, fov, segm_fld = r'Y:\\RenImaging2\\Brett\\CellCulture_EGFR_03_01_2025_Segmentation\\', 
                pix_size = [0.5,0.108333,0.108333], save = False, save_fld = r'Y:\RenImaging2\Brett\CellCulture_EGFR_03_01_2025_DNA_Analysis'):
    ## load segm file
    ## this is specific to the coCulture naming convention
    set_ = '_set' + fov.split('_')[1].split('zscan')[-1]
    ifov = int(fov.split('_')[-1])
    segm_fl = os.path.join(segm_fld, f'{fov}--H1_P1F123{set_}--dapi_segm.npz')
    segm = np.load(segm_fl)['segm']
    shape =  np.load(segm_fl)['shape']
    resc = shape / segm.shape
    
    ## get volm per cell
    Xcells = np.array(np.where(segm>0)).T
    icells = segm[tuple(Xcells.T)]
    X_cells = {icell:Xcells[icells==icell]for icell in np.unique(icells)}
    
    info_rgs = []
    for icell in np.unique(XF_col[:,-1]):
        icell = int(icell)
        Xcells = X_cells[icell]
        volm = len(Xcells)
        keep_cell = XF_col[:,-1]==icell
        necdna = np.sum(keep_cell)
        iXcells = np.arange(len(Xcells))
        ## denom: rgs of randomly sample necdna points
        rgsim = [get_rgs(Xcells[np.random.choice(iXcells,necdna)],pix_size = pix_size * resc) for i in np.arange(100)]
        rgsim = np.mean(rgsim)
        ## numerator: rgs of bright spots
        rg = get_rgs(XF_col[keep_cell][:,:3],pix_size = pix_size)
        norm_rg = rg/rgsim
        
        ifovcell = int(ifov * 10**5 + icell)
        info_rgs.append([ifovcell,necdna,volm,rg,norm_rg])
    
    if save:
        save_fl = os.path.join(save_fld, f'{fov}--rgs.npz')
        np.savez_compressed(save_fl, info_rgs = info_rgs, header = ['ifovcell','necdna','volm','rg','norm_rg'], XF_col = XF_col)
        print(f'rgs saved to {save_fl}')
    return info_rgs

###### ---------------------------
###### select FOVs (RNA)
###### ---------------------------

def get_R_RNA(fl):
    htag = get_htag(fl)
    hindex = get_hindex(fl)
    icol = get_icol(fl)
    if hindex == 1:
        Rs = [1,2,3]
        return Rs[icol]
    else:
        return int(hindex)*1000 + icol
def is_RNA(fl): return ('H' in get_htag(fl)) and (get_hindex(fl) > 1)

def get_Xh_fov_RNA(fov,  
               # set_ = None,
               segm_fld = r'C:\GBM39\TK_FFBBL1_SampleA_1_27_2023__JennySegmentation\merged_segmentation', 
                analysis_folder=r'C:\GBM39\TK_FFBBL1_SampleA_1_27_2023__JennyAnalysis',
                save_fld_traces=r'C:\GBM39\ecDNATracer\fulldata', 
                   force = False, smFISH = False,
              pix_size = [0.3, 0.108333, 0.108333], save = False):
    """
    Processes a Field of View (FOV) for drift correction, chromosome alignment, and cell segmentation.
    
    Parameters:
        ifov (int): Index of the FOV to process.
        analysis_folder (str, optional): Path to the analysis folder. Defaults to provided path.
        save_fld_traces (str, optional): Path to save the processed results. Defaults to provided path.
        smFISH (boolean): allow for singel round processing
    """

    # Load FOVs
    # if set_ is None:
    #     fovs = np.load(os.path.join(analysis_folder, 'fovs__.npy'))
    #     fov = fovs[ifov].replace('.zarr', '')
    # else:
    #     fovs = np.load(os.path.join(analysis_folder, f'fovs___set{set_}.npy'))
    #     fov = fovs[ifov].replace('.zarr', '')

    print(f'Processing FOV = {fov}...')
    
    save_fl_traces = os.path.join(save_fld_traces, f"{fov}--XF_RNA.npz")
    if os.path.exists(save_fl_traces):
        if not force:
            print(f'{save_fl_traces} exists already...')
            return
    
    # get all segm files
    segm_fls = glob.glob(os.path.join(segm_fld, '*segm.npz'))
    dic_segm_fls = dict(zip([os.path.basename(fl).split('--')[0] for fl in segm_fls], segm_fls))
    segm_merged_fl = dic_segm_fls[fov]

    # Find files
    fits_fls = glob.glob(os.path.join(analysis_folder, '*__Xhfits.npz'))
    drift_fls = glob.glob(os.path.join(analysis_folder, 'drift*'))
    # print(fits_fls)

    # Identify drift file for this FOV
    drift_fl = [fl for fl in drift_fls if fov in fl]
    if not drift_fl:
        if not smFISH:
            print(f"No drift file found for FOV: {fov}")
            return
        else: ##smFISH
            dic_drifts = None
    else:
        drift_fl = drift_fl[0]
        # Load drift data
        drifts, flds, fov_, fl_ref = np.load(drift_fl, allow_pickle=True)
        dic_drifts = {os.path.basename(fld): drft[0] for fld, drft in zip(flds, drifts)}
    
    # print(dic_drifts)

    # Filter fits files
    fits_fov = [fl for fl in fits_fls if fov in fl]
    fits_chr = [fl for fl in fits_fov if is_RNA(fl)]  # is_chr needs to be defined
    # print(fits_fov)
    # print(fits_chr)
    
    # Sort and create dictionary
    ordered_fits_chr = np.array(fits_chr)[np.argsort([get_hindex(fl) for fl in fits_chr])]
    dic_fits_RNA = {get_R_RNA(fl): fl for fl in ordered_fits_chr} 
    Rs = np.sort(list(dic_fits_RNA.keys()))

    
    # Load segmentation files
    if 'cyto_segm' in np.load(segm_merged_fl).keys():
        im_segm = np.load(segm_merged_fl)['cyto_segm']
        im_segmNuc = np.load(segm_merged_fl)['dapi_segm']
        shape = np.load(segm_merged_fl)['shape']
        resc = np.round(shape / im_segm.shape).astype(int)
        im_segm_ = im_segm  # Optionally expand segmentation
    else: ## this is with dapi segm only
        im_segm = np.load(segm_merged_fl)['segm']
        im_segmNuc = np.load(segm_merged_fl)['segm']
        shape = np.load(segm_merged_fl)['shape']
        resc = np.round(shape / im_segm.shape).astype(int)
        im_segm_ = im_segm  # Optionally expand segmentation

    # Get drift-corrected matrix
    XH = get_XH(dic_fits_RNA, dic_drifts, th_h=0, th_cor=0.25, chrom_fl = None)  # get_XH needs to be defined
    # print(dic_fits_RNA)
    # print(XH.shape)

    # Assign to cells
    icells = np.zeros(len(XH))
    Xred = np.round(XH[:, :3] / resc).astype(int)
    keep = np.all((Xred < im_segm_.shape) & (Xred >= 0), axis=-1)
    icells[keep] = im_segm_[tuple(Xred[keep].T)]

    # Identify nuclear markers
    isnuc = np.zeros(len(XH))
    isnuc[keep] = im_segmNuc[tuple(Xred[keep].T)] > 0

    # Concatenate data
    XF = np.concatenate([XH[icells > 0], isnuc[icells > 0][:, np.newaxis], icells[icells > 0][:, np.newaxis]], axis=-1)
    
    ## swap the order such that
    ## -5: isNuc, -4: H, -3: color, -2: R, -1: cell
    num_cols = XF.shape[1]
    old_idx = num_cols - 2  # -2nd column
    new_idx = num_cols - 5  # -5th position
    # Get column indices
    col_indices = np.arange(num_cols)
    # Remove the -2nd column
    col_indices = np.delete(col_indices, old_idx)
    # Insert the column at the -5th position
    col_indices = np.insert(col_indices, new_idx, old_idx)
    XF_ = XF[:, col_indices]
    header = ['z','x','y','bk','corabs','habs','cor','isNuc', 'h','col','R', 'cell_id']
    dic_header = dict(zip(header, range(len(header))))
    
    # Save processed data
    if save:
        os.makedirs(save_fld_traces, exist_ok=True)    
        np.savez(save_fl_traces, XF=XF_, pix_size=pix_size, header = header)
        print(f"Saved processed data for FOV {fov} at {save_fl_traces}")
    
    return XF_

def get_counts_per_fov(fov, XF_RNA,
                       save_fld,
                              # genesofInterestR = [dic_genes_R['VIM'], dic_genes_R['NEFL']], 
                       genesofInterestR = None,
                       th_h = 3,
                       save = False, plot_distr = False):
    
    save_fl = os.path.join(save_fld, 'RNA_analysis', f'{fov}--XF_cts.npz')
    if os.path.exists(save_fl):
        print(f'{save_fl} exists...')
    ## select brighter spots
    XF_RNA_b = select_britgher_spots(XF_RNA,medh, uRs, th_h=th_h)
    
    ## get ufovcell
    ifov = int(fov.split('_')[-1])
    
    ## ucells should be all cells, not only the ones with cts from two rounds
    ucells_all = np.unique(XF_RNA_b[:, -1])
    ufovcells = [int(ifov * 10**5 + icell) for icell in ucells_all]
    
    ## get counts per cell for genes of interest
    if genesofInterestR is not None:
        ucells, cts0, cts1 = plot_cts_distr(XF_RNA_b, genesofInterestR, plot = plot_distr)
        np.savez_compressed(save_fl, XFb = XF_RNA_b, ucells = ucells, ufovcells = ufovcells,
                            cts0 = cts0, cts1 = cts1, genes = [dic_R_genes[R] for R in genesofInterestR])
        return ufovcells, cts0, cts1
    else:
        np.savez_compressed(save_fl, XFb = XF_RNA_b, ucells = ucells_all, ufovcells = ufovcells)
        return XF_RNA_b
    
    
def get_gene_cts_per_cell(XF_all, geneR, dic_header = dic_header, nuc_only = False):
    XF_gene = XF_all[XF_all[:, dic_header['R']] == geneR]
    if nuc_only:
        XF_gene = XF_gene[XF_gene[:, dic_header['isNuc']] == 1]
    ucells, cts_cells = np.unique(XF_gene[:, dic_header['cell_id']], return_counts = True)
    dic_ucells_cts = dict(zip(ucells, cts_cells))
    return dic_ucells_cts