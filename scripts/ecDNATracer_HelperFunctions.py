import glob,os,numpy as np
from tqdm import tqdm
import pickle
def get_XH_colorflip(dic_fits_chr,dic_drifts,th_h=0,th_cor=0.4,chrm_fl=None):
    XH = []
    for R in tqdm(list(dic_fits_chr.keys())):
        save_fl = dic_fits_chr[R]
        icol = get_icol(save_fl)
        Xh = np.load(save_fl,allow_pickle=True)['Xh']
        if len(Xh.shape):
            Xh = Xh[Xh[:,-1]>th_h]
            Xh = Xh[Xh[:,-2]>th_cor]
            if len(Xh):
                tzxy = dic_drifts[get_htag(save_fl)]
                ## correct for color flip
                icol_,R_ = flip_P4_color(icol, R)
                ### apply chrom abberation
                if chrm_fl is not None:
                    Ms = pickle.load(open(chrm_fl,'rb'))
                    M = Ms[icol_]
                    Xh[:,:3] = apply_colorcor(Xh[:,:3],m=M)
                
                Xh[:,:3]+=tzxy# drift correction
                
                icolR = np.array([[icol_,R_]]*len(Xh))
                # icolR = np.array([[icol,R]]*len(Xh))
                XH_ = np.concatenate([Xh,icolR],axis=-1)
                XH.extend(XH_)
    XH = np.array(XH)
    return XH
def flip_P4_color(icol, R):
    import pickle
    fl = "C:/GBM39/design/FFBBDNA_252color-dic_hyb_color.pkl"
    with open(fl, 'rb') as file:
        dic_hyb_color = pickle.load(file)
    icol_ = dic_hyb_color[R]
    if icol_ != icol: print(f'color mismatch detected in R{R}')
    
    return icol_, R
def get_dic_fits_chr(ordered_fits_chr, ifov = None):
    # get fov from file
    if ifov is None:
        ifov = int(os.path.basename(ordered_fits_chr[0]).split('--')[0].split('__')[-1])
    ## use later batch for ifov > 50
    if ifov >=50:
        dic_fits_chr = {get_R(fl): fl for fl in ordered_fits_chr}
    else:
        ## use first batch for the first FOVs
        dic_fits_chr = {}
        for fl in ordered_fits_chr:
            R = get_R(fl)
            if R in dic_fits_chr.keys():
                if (R < 19) & (R > 0): continue ### keep the first round for R1-18
            dic_fits_chr[R] = fl
    return dic_fits_chr
def select_britgher_spots(XF,medh,uRs,th_h=5):
    ths = medh[:,1]+th_h*medh[:,2]
    ths_ = np.zeros(np.max(uRs)+1)
    ths_[uRs]=ths
    H = XF[:,-5]
    Rs= XF[:,-3].astype(int)
    keeph = H>ths_[Rs]
    XF_ = XF[keeph]
    return XF_
def apply_chromatic_correction_per_cell(XF,medh,uRs,th_h=5,th_d = 5):
    XF_ = select_britgher_spots(XF,medh,uRs,th_h=th_h)
    cells_ = XF_[:,-1].astype(int)
    ucells_ = np.unique(cells_)

    XF_col = XF.copy()
    Col_col = XF_col[:,-4].astype(int)
    cells_col = XF_col[:,-1].astype(int)

    for ucell in ucells_:
        XF_cell = XF_[cells_==ucell]
        X = XF_cell[:,:3]
        R = XF_cell[:,-3].astype(int)
        Col = XF_cell[:,-4].astype(int)

        from scipy.spatial import KDTree
        tree = KDTree(X)
        pairs = tree.query_pairs(th_d,output_type='ndarray')
        R1,R2 = R[pairs].T
        pairs = pairs[np.abs(R1-R2)==1]

        #ref_col,new_col = (0,2)
        for ref_col,new_col in [(1,0),(1,2)]:
            pairs01 = pairs[np.all(Col[pairs]==[ref_col,new_col],axis=1)]
            pairs10 = pairs[np.all(Col[pairs]==[new_col,ref_col],axis=1)]
            pairs01f = np.concatenate([pairs01,pairs10[:,::-1]])
            X1,X2 = X[pairs01f].swapaxes(0,1)
            tzxy = np.mean(X2-X1,axis=0)
            #print(tzxy)
            keep = (Col_col==new_col)&(cells_col==ucell)
            Xf_cell_col = XF_col[keep]
            Xf_cell_col[:,:3]=Xf_cell_col[:,:3]-tzxy
            XF_col[keep]=Xf_cell_col
    return XF_col


def get_htag(fl): return os.path.basename(fl).split('--')[1]
def is_chr(fl): return ('R' in get_htag(fl)) and ('_' in get_htag(fl)) # check if chromatin data of type H*R, different than RNA data which is H*Q* -exons or H*I* - introns
def get_hindex(fl): return int(get_htag(fl).replace('_','').split('R')[0][1:])
def get_icol(fl): return int(fl.split('--col')[-1].split('_')[0])
def get_R(fl):
    htag = get_htag(fl)
    Rs = np.array(htag.split('R')[-1].split('_'),dtype=int)
    icol = get_icol(fl)
    return Rs[icol]
def get_XH(dic_fits_chr,dic_drifts,th_h=0,th_cor=0.4):
    XH = []
    for R in tqdm(list(dic_fits_chr.keys())):
        save_fl = dic_fits_chr[R]
        icol = get_icol(save_fl)
        Xh = np.load(save_fl,allow_pickle=True)['Xh']
        if len(Xh.shape):
            Xh = Xh[Xh[:,-1]>th_h]
            Xh = Xh[Xh[:,-2]>th_cor]
            if len(Xh):
                tzxy = dic_drifts[get_htag(save_fl)]
                Xh[:,:3]+=tzxy# drift correction
                icolR = np.array([[icol,R]]*len(Xh))
                XH_ = np.concatenate([Xh,icolR],axis=-1)
                XH.extend(XH_)
    XH = np.array(XH)
    return XH
def expand_segmentation(im_segm_,nexpand=4):
    from scipy import ndimage as nd
    im_segm__ = im_segm_.copy()
    im_bw = im_segm_>0
    im_bwe = nd.binary_dilation(im_bw,iterations=nexpand)
    Xe = np.array(np.where(im_bwe>0)).T
    X = np.array(np.where(im_bw>0)).T
    Cell = im_segm_[tuple(X.T)]
    from scipy.spatial import KDTree
    inds = KDTree(X).query(Xe)[-1]
    im_segm__[tuple(Xe.T)]=Cell[inds]
    return im_segm__
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
def get_chrom_mat(XF,medh,uRs,th_h=5,th_d = 0.5,pix_size=[0.25,0.10833,0.10833]):
    ths = medh[:,1]+th_h*medh[:,2]
    ths_ = np.zeros(np.max(uRs)+1)
    ths_[uRs]=ths
    H = XF[:,-4]
    Rs= XF[:,-2].astype(int)
    keeph = H>ths_[Rs]
    XF_ = XF[keeph]
    #np.unique(cells[keeph],return_counts=True)[-1]/len(uRs)

    cells_ = XF_[:,-1].astype(int)
    ucells_ = np.unique(cells_)
    Rs_ = XF_[:,-2].astype(int)
    uRs_ = np.unique(Rs_)

    difs = {ucell:[] for ucell in ucells_}
    idelta=1

    for uR in tqdm(uRs_[:-idelta]):
        if uR%3==1:
            isR1 = (Rs_==uR)
            isR2 = (Rs_==(uR+idelta))
            XF__1 = XF_[isR1]
            XF__2 = XF_[isR2]
            for ucell in ucells_:
                X__1 = XF__1[ucell==cells_[isR1]][:,:3]
                X__2 = XF__2[ucell==cells_[isR2]][:,:3]
                if len(X__1)>0 and len(X__2)>0:

                    from scipy.spatial import KDTree
                    dists,inds = KDTree(X__1*pix_size).query(X__2*pix_size)
                    difs[ucell].extend((X__2-X__1[inds])[dists<th_d])
    #chrom_mat = [[np.mean(difs[icell],axis=0) for icell in difs]]
    chrom_mat = [[np.mean(difs[icell], axis=0) if difs[icell] else np.array([np.nan, np.nan, np.nan]) for icell in difs]]
    difs = {ucell:[] for ucell in ucells_}
    for uR in tqdm(uRs_[:-idelta]):
        if uR%3==2:
            isR1 = (Rs_==uR)
            isR2 = (Rs_==(uR+idelta))
            XF__1 = XF_[isR1]
            XF__2 = XF_[isR2]
            for ucell in ucells_:
                X__1 = XF__1[ucell==cells_[isR1]][:,:3]
                X__2 = XF__2[ucell==cells_[isR2]][:,:3]
                if len(X__1)>0 and len(X__2)>0:
                    pix_size=[0.25,0.10833,0.10833]
                    from scipy.spatial import KDTree
                    dists,inds = KDTree(X__1*pix_size).query(X__2*pix_size)
                    difs[ucell].extend((X__2-X__1[inds])[dists<th_d])
    # return chrom_mat, difs
    chrom_mat+=[np.array(chrom_mat[0])*0]
    # chrom_mat+=[np.zeros((len(chrom_mat[0]), 3))]
    # chrom_mat+=[[-np.mean(difs[icell],axis=0) for icell in difs]]
    chrom_mat+=[[-np.mean(difs[icell],axis=0) if difs[icell] else np.array([np.nan, np.nan, np.nan]) for icell in difs]]

    chrom_mat=np.array(chrom_mat)
    return chrom_mat, XF_

def get_XF_chr(XF,medh,uRs,chrom_mat_,th_h=3):
    ths = medh[:,1]+th_h*medh[:,2]
    ths_ = np.zeros(np.max(uRs)+1)
    ths_[uRs]=ths
    H = XF[:,-4]
    Rs = XF[:,-2].astype(int)
    keeph = H>ths_[Rs]
    XF_ = XF[keeph]

    cells_ = XF_[:,-1].astype(int)
    ucells_ = np.unique(cells_)
    Rs_ = XF_[:,-2].astype(int)
    uRs_ = np.unique(Rs_)
    cols_ = XF_[:,-3].astype(int)
    ucols_ = np.unique(cols_)

    XF_chr = XF_.copy()
    # adjust cell index
    dic_index_map = dict(zip(ucells_, range(len(ucells_))))
    cells__ = [dic_index_map[x] for x in cells_]
    XF_chr[:,:3]=XF_chr[:,:3]+chrom_mat_[cols_,cells__,:]
    XF_chr_ = XF_chr[~np.any(np.isnan(XF_chr), axis=1)]
    return XF_chr_
def calc_color_matrix(x,y,order=2):
    """This gives a quadratic color transformation (in matrix form)
    x is Nx3 vector of positions in the reference channel (typically cy5)
    y is the Nx3 vector of positions in another channel (i.e. cy7)
    return m_ a 3x7 matrix which when multipled with x,x**2,1 returns y-x
    This m_ is indended to be used with apply_colorcor
    """ 
    x_ = np.array(y)# ref zxy
    y_ = np.array(x)-x_# dif zxy
    # get a list of exponents
    exps = []
    for p in range(order+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    # construct A matrix
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    m_ = [np.linalg.lstsq(A, y_[:,iy])[0] for iy in range(len(x_[0]))]
    m_=np.array(m_)
    return m_
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


#### Helper function to process input (update 071025)
### Taken from `063025_run_rough_traces.ipynb`
import re
def get_XH_colorflip_chromabb(dic_fits_chr,dic_drifts,th_h=0,th_cor=0.4,chrm_fl=None,
                              dic_hyb_col_fl = "C:/GBM39/design/FFBBDNA_252color-dic_hyb_color.pkl",
                             Rmin = 1, Rmax = 252):
    XH = []
    for R in tqdm(list(dic_fits_chr.keys())):
        if (R > Rmax) or (R < Rmin):
            continue
        save_fl = dic_fits_chr[R]
        ## get the correct color here
        import pickle
        with open(dic_hyb_col_fl, 'rb') as file:
            dic_hyb_color = pickle.load(file)
        icol = dic_hyb_color[R]#get_icol(save_fl)
        save_fl = re.sub(r'--col\d', f'--col{icol}', save_fl)
        Xh = np.load(save_fl,allow_pickle=True)['Xh']
        # print(save_fl)
        # print(rf'R = {R}, XH ={len(Xh)}')
        # if len(Xh.shape) & (len(Xh) > 0):
        if len(Xh) > 0:
            Xh = Xh[Xh[:,-1]>th_h]
            Xh = Xh[Xh[:,-2]>th_cor]
            if len(Xh):
                tzxy = dic_drifts[get_htag(save_fl)]
                ### apply chrom abberation
                if chrm_fl is not None:
                    Ms = pickle.load(open(chrm_fl,'rb'))['Ms']
                    M = Ms[icol]
                    Xh[:,:3] = apply_colorcor(Xh[:,:3],m=M)
                
                Xh[:,:3]+=tzxy# drift correction
                
                # icolR = np.array([[icol_,R_]]*len(Xh))
                icolR = np.array([[icol,R]]*len(Xh))
                XH_ = np.concatenate([Xh,icolR],axis=-1)
                XH.extend(XH_)
    XH = np.array(XH)
    return XH

def prepare_input_fov_chromabb(fov,
                         analysis_folder=r'C:\GBM39\TK_FFBBL1_SampleA_1_27_2023__JennyAnalysis',
    save_fld_traces=r'D:\GBM39\ecDNATracer\fulldataBB',
    segm_folder = r'C:\GBM39\TK_FFBBL1_SampleA_1_27_2023__JennySegmentation\merged_segmentation',
    pix_size = [0.25, 0.108333, 0.108333],
                              chrm_fl = r"C:\GBM39\ecDNATracer\dic_chromatic_abberation_Jenny_Scope4_07102025.pkl",
                               dic_hyb_col_fl = "C:/GBM39/design/FFBBDNA_252color-dic_hyb_color.pkl",
                              Rmin = 1, Rmax = 252, save = True, dapi_segm_only = False):
    
    os.makedirs(save_fld_traces, exist_ok=True)
    save_fl_traces = os.path.join(save_fld_traces, f"{fov}_colorflip_chromabb--XF.npz")
    
    # get all segm files
    if dapi_segm_only:
        segm_fls = glob.glob(os.path.join(segm_folder, '*--dapi_segm.npz'))
    else:
        segm_fls = glob.glob(os.path.join(segm_folder, '*--dapi-cyto_segm.npz'))
    dic_segm_fls = dict(zip([os.path.basename(fl).split('--')[0] for fl in segm_fls], segm_fls))
    segm_merged_fl = dic_segm_fls[fov]
    
    ###################################################
#     # Find files
#     fits_fls = glob.glob(os.path.join(analysis_folder, f'*__Xhfits.npz'))
#     drift_fls = glob.glob(os.path.join(analysis_folder, 'drift*'))

#     # Identify drift file for this FOV
#     drift_fl = [fl for fl in drift_fls if fov in fl]
#     drift_fl

#     drift_fl = drift_fl[0]


#     # Filter fits files
#     fits_fov = [fl for fl in fits_fls if fov in fl]
#     fits_chr = [fl for fl in fits_fov if is_chr(fl)]
    ##################################################
    # Direct glob for fits just for this FOV
    fits_fov = glob.glob(os.path.join(analysis_folder, f'{fov}*__Xhfits.npz'))
    fits_chr = [fl for fl in fits_fov if is_chr(fl)]

    # Direct glob for drift file
    drift_match = glob.glob(os.path.join(analysis_folder, f'drift*{fov}*'))
    if not drift_match:
        raise FileNotFoundError(f"No drift file found for FOV: {fov}")
    drift_fl = drift_match[0]


    # Sort and create dictionary
    ordered_fits_chr = np.array(fits_chr)[np.argsort([get_hindex(fl) for fl in fits_chr])]
    # dic_fits_chr = {get_R(fl): fl for fl in ordered_fits_chr} 
    ## TODO: use high quality read FOV < 50
    dic_fits_chr = get_dic_fits_chr(ordered_fits_chr, None)
    Rs = np.sort(list(dic_fits_chr.keys()))


    # Load drift data
    drifts, flds, fov_, fl_ref = np.load(drift_fl, allow_pickle=True)
    dic_drifts = {os.path.basename(fld): drft[0] for fld, drft in zip(flds, drifts)}

    # Load segmentation files
    if dapi_segm_only:
        im_segm = np.load(segm_merged_fl)['segm']
        im_segmNuc = np.load(segm_merged_fl)['segm']
    else:
        im_segm = np.load(segm_merged_fl)['cyto_segm']
        im_segmNuc = np.load(segm_merged_fl)['dapi_segm']
    shape = np.load(segm_merged_fl)['shape']
    resc = np.round(shape / im_segm.shape).astype(int)
    im_segm_ = im_segm  # Optionally expand segmentation
    # Get drift and chrom abb corrected matrix
    # XH = get_XH_colorflip(dic_fits_chr, dic_drifts, th_h=0, th_cor=0,chrm_fl = None)
    XH = get_XH_colorflip_chromabb(dic_fits_chr, dic_drifts, th_h=0, th_cor=0.25, 
                                   chrm_fl=chrm_fl, Rmin = Rmin, Rmax = Rmax, dic_hyb_col_fl = dic_hyb_col_fl)

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

    # Save processed data
    if save:
        np.savez(save_fl_traces, XF=XF, pix_size=pix_size)
    return XF

def get_maxH_for_XF(XF, subsample = 50):
    cells = XF[:,-1].astype(int)
    ucells = np.unique(cells)
    if len(ucells) < subsample:
        subsample = len(ucells)
    ucells_sub = np.random.choice(ucells,size = subsample, replace = False)
    Rs = XF[:,-3].astype(int)
    uRs = np.unique(Rs)
    cols = XF[:,-4].astype(int)
    ucols = np.unique(cols)
    maxH = []
    for uR in tqdm(uRs):
        isR = (Rs==uR)
        XF_ = XF[isR]
        maxH_ = []
        for ucell in ucells_sub:
            iscell = cells[isR]==ucell
            if np.sum(iscell):
                h_ = XF_[iscell][:,-5]
                h__ = np.sort(h_)
                max_ = np.median(h__[-5:])
                min_ = np.median(h__[:5])
                std_ = np.std(h__[:])
                maxH_.append([max_,min_,std_])
        maxH.append(maxH_)
    maxH = np.array(maxH)
    medh = np.median(maxH,axis=1)
    return medh,uRs

def get_traces_napari(traces_cell, default_colors = None):
    Xtraces = []
    cols = []
    for i, trace in enumerate(traces_cell):
        Xtrace = trace[:,:3].copy()
        Xtrace = Xtrace[~np.isnan(Xtrace[:,0])]
        Xtraces.append(Xtrace)
        if default_colors is None:
            color = np.random.random(3)
        else:
            color = default_colors[i % len(default_colors)]  # wrap around if needed
        cols.append([color] * len(Xtrace))
    Xtraces = np.concatenate(Xtraces)
    cols = np.concatenate(cols)
    return Xtraces,cols


#### Helper functions for identifying chrom copy
#### taken from GBM39/ecDNATracer/071425_linear_trace_analysis.ipynb
import numpy as np
from scipy.spatial import cKDTree

def remove_close_points(XH, dth=3):
    """
    Remove points from XH that are within Euclidean distance dth of each other,
    computed only on the first three columns (z, x, y). Keeps the first in each cluster.

    Parameters:
    - XH: numpy array of shape (N, D), where D >= 3 and first three columns are coordinates
    - dth: float, minimum allowed distance between any two points

    Returns:
    - XH_filtered: numpy array of filtered points (all columns retained)
    """
    coords = XH[:, :3]
    tree = cKDTree(coords)
    N = len(coords)
    keep = np.ones(N, dtype=bool)

    for i in range(N):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], dth)
        for j in neighbors:
            if j > i:
                keep[j] = False

    return XH[keep]


def get_bright_points_per_R(XF, iR, medh, uRs,
                            th_hlow = 6, ## std above the min
                            th_dd = 3, H_col = -5):
    '''
    return XH0_bright, XH1_bright, and Xmatch_avg (average X of colocalized points)
    For bright spot selection, using a std approach
    '''
    ## brightness threshold

    ths = medh[:,1]+th_hlow*medh[:,2]
    ths_ = np.zeros(np.max(uRs)+1)
    ths_[uRs]=ths
    H = XF[:,H_col]
    Rs= XF[:,-3].astype(int)
    keeph = H>ths_[Rs]
    XF_ = XF[keeph]

    ## keep only that iR
    XH = XF_[XF_[:, -3] == iR]

    ## being more strict
    ## per color, no points within 3 pixels can be selected twice
    XH_ = remove_close_points(XH, dth = th_dd)

    
    return XH_

def get_anchor_points_per_fov(ifovcell, XF_downstream, XF_del,
                             medh, uRs, 
                              th_hlow = 10, th_dd = 5,
                             save_fld = r'D:\GBM39\ecDNATracer\gridSearch_tracesBB_chromabb\anchor_points-iR73_253_254', force = False):
    
    save_fl = os.path.join(save_fld, f'cell{int(ifovcell)}-th_h{th_hlow}-anchor_points-iR73_253_254.npz')
    if not os.path.exists(save_fl) or force:

        ## downstream
        X253 = get_bright_points_per_R(XF_downstream, iR = 253, medh = medh, uRs = uRs,
                              th_hlow = th_hlow, ## std above the min
                            th_dd = th_dd) ## distance in pixel

        X254 = get_bright_points_per_R(XF_downstream, iR = 254, medh = medh, uRs = uRs,
                              th_hlow = th_hlow, ## std above the min
                            th_dd = th_dd) ## distance in pixel

        X73 = get_bright_points_per_R(XF_del, iR = 73, medh = medh, uRs = uRs,
                              th_hlow = th_hlow, ## std above the min
                            th_dd = th_dd) ## distance in pixel

        ## save it for each cell
        np.savez(save_fl, X253 = X253, X254 = X254, X73 = X73)
        return X73, X253, X254
    else:
        X73 = np.load(save_fl)['X73']
        X253 = np.load(save_fl)['X253']
        X254 = np.load(save_fl)['X254']
        return X73, X253, X254
    
from scipy.spatial import cKDTree
import numpy as np

def filter_XTraces_by_distance_only(XTraces, Xmatch, dth=1.0, pix_size=[0.25, 0.108, 0.108]):
    """
    Filters XTraces to retain only those traces that have at least one point
    spatially close to any point in Xmatch (ignores genomic R distance).

    Parameters:
    - XTraces: list of (N x 7) numpy arrays
    - Xmatch: (M x 12) numpy array
    - dth: float, spatial distance threshold (in microns)
    - pix_size: list of 3 floats, voxel size in [z, y, x] to convert pixel to micron

    Returns:
    - XTraces_nearby: list of filtered traces from XTraces
    - iXTraces_nearby: list of indices into original XTraces
    """
    XTraces_nearby = []
    iXTraces_nearby = []

    # Convert Xmatch spatial coordinates to microns
    Xmatch_um = Xmatch.copy()
    Xmatch_um[:, 0] *= pix_size[0]  # z
    Xmatch_um[:, 1] *= pix_size[1]  # x
    Xmatch_um[:, 2] *= pix_size[2]  # y

    tree = cKDTree(Xmatch_um[:, :3])

    for ix, X_um in enumerate(XTraces):
        if len(X_um) == 0:
            continue

        dists, _ = tree.query(X_um[:, :3], distance_upper_bound=dth)

        if np.any(np.isfinite(dists)):
            XTraces_nearby.append(X_um)
            iXTraces_nearby.append(ix)

    return XTraces_nearby, iXTraces_nearby

def summarize_chrom_intersections(iX_253, iX_254, iX_73, ifovcell):
    """
    Computes lengths, intersections, and unions of index sets. Returns counts with descriptions.

    Parameters:
    - iX_253, iX_254, iX_73: lists of indices
    - ifovcell: identifier (int or str) for the current cell/FOV

    Returns:
    - chrom_cts: list of counts, starting with ifovcell
    - header: list of labels for each count
    """
    s253 = set(iX_253)
    s254 = set(iX_254)
    s73 = set(iX_73)

    chrom_cts = [ifovcell]
    header = ["ifovcell"]

    # Individual lengths
    chrom_cts.extend([len(s253), len(s254), len(s73)])
    header.extend(["len_253", "len_254", "len_73"])

    # Intersection of all three
    chrom_cts.append(len(s253 & s254 & s73))
    header.append("intersection_253_254_73")

    # Pairwise intersections
    chrom_cts.append(len(s253 & s254))
    header.append("intersection_253_254")

    chrom_cts.append(len(s253 & s73))
    header.append("intersection_253_73")

    chrom_cts.append(len(s254 & s73))
    header.append("intersection_254_73")

    # Union of 253 and 254
    chrom_cts.append(len(s253 | s254))
    header.append("union_253_254")

    # Union of all three
    chrom_cts.append(len(s253 | s254 | s73))
    header.append("union_all")

    return chrom_cts, header

import numpy as np

def filter_circular_traces(traces_linear, traces_all_circular, th_overlap=0.1, verbose = False):
    """
    Removes circular traces that overlap too much with any linear trace.
    
    Parameters:
    - traces_linear: np.ndarray of shape (N1, T, D), the reference set
    - traces_all_circular: np.ndarray of shape (N2, T, D), the set to filter
    - th_overlap: float, threshold of overlap above which a circular trace is removed
    
    Returns:
    - traces_circular: np.ndarray of shape (<=N2, T, D), post-filtered circular traces
    """
    def overlap_fraction(tr1, tr2):
        valid = ~(np.isnan(tr1[:, 0]) & np.isnan(tr2[:, 0]))
        if np.sum(valid) == 0:
            return 0
        return np.sum(np.all(tr1 == tr2, axis=-1)) / np.sum(valid)
    
    keep_indices = []
    for i, tr_circ in enumerate(traces_all_circular):
        is_duplicate = False
        for tr_lin in traces_linear:
            fr = overlap_fraction(tr_circ, tr_lin)
            if fr > th_overlap:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)
    
    traces_circular = traces_all_circular[keep_indices]
    if verbose:
        print(f"Kept {len(traces_circular)} of {len(traces_all_circular)} circular traces")
    return traces_circular

#### Helper functions for plotting analysis
#### taken from GBM39/invirto_030125/TracerResult.ipynb
def compute_med_dist_mat(traces, contact_dist = 0.25):
    '''
    If contact_dist is not None, it returns the mean contact matrix
    Otherwise, it returns the M x 252 x 252 dist matrix
    '''
    from scipy.spatial.distance import pdist,squareform
    mats = []
    for tr in tqdm(traces):
        X = tr[:,:3]
        mats.append(squareform(pdist(X)))
    mats = np.array(mats)

    if contact_dist is not None:
        mats = np.mean(mats<contact_dist,axis=0)/np.mean(mats>=0.,axis=0)

    return mats

def plot_linear_vs_circular_subpanels_with_type(linear_traces, circ_traces, linear_type, plot_type, 
                                                subsample = False, vranges = None):
    ncols = 3
    nrows = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    if subsample:
        circ_traces = circ_traces[np.random.choice(range(len(circ_traces)), size = len(linear_traces), replace = False), :, :]
    ## linear
    ax = axes[0]
    if plot_type == 'dist':
        if vranges is None:
            vmin = 0.6
            vmax = 1.2
        else:
            vmin,vmax = vranges
        mat_linear = np.nanmedian(compute_med_dist_mat(linear_traces, contact_dist = None), axis = 0)
        im = ax.imshow(mat_linear, cmap='seismic_r', interpolation='nearest', vmin = vmin, vmax=vmax)
    else:
        if vranges is None:
            vmin = -5
            vmax = -2
        else:
            vmin,vmax = vranges
            
        mat_linear = np.log(compute_med_dist_mat(linear_traces, contact_dist = 0.25))
        im = ax.imshow(mat_linear, cmap='seismic', interpolation='nearest',vmin = vmin, vmax=vmax)

    ax.set_title(f'{len(np.unique(linear_traces[:, :, -1]))} cells, {len(linear_traces)} linear traces\n {linear_type}')
    fig.colorbar(im, ax=ax, shrink=0.6)
    
    ## circular
    ax = axes[1]
    if plot_type == 'dist':
        if vranges is None:
            vmin = 0.6
            vmax = 1.2
        else:
            vmin,vmax = vranges
        mat_circular = np.nanmedian(compute_med_dist_mat(circ_traces, contact_dist = None), axis = 0)
        im = ax.imshow(mat_circular, cmap='seismic_r', interpolation='nearest', vmin = vmin, vmax=vmax)
    else:
        if vranges is None:
            vmin = -5
            vmax = -2
        else:
            vmin,vmax = vranges
        mat_circular = np.log(compute_med_dist_mat(circ_traces, contact_dist = 0.25))
        im = ax.imshow(mat_circular, cmap='seismic', interpolation='nearest', vmin = vmin, vmax=vmax)

    ax.set_title(f'{len(np.unique(circ_traces[:, :, -1]))} cells, {len(circ_traces)} circular traces')
    fig.colorbar(im, ax=ax, shrink=0.6)
    
    ## diff
    diff_mat = mat_linear - mat_circular

    ax = axes[2]
    im = ax.imshow(diff_mat,vmin=-0.5,vmax=0.5,cmap='seismic',interpolation='nearest')
    ax.set_title(f'Diff in {plot_type}: linear - circular')
    fig.colorbar(im, ax=ax, shrink=0.6)
    

    # # Hide any unused axes
    # for j in range(i + 1, len(axes)):
    #     axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def compare_distance_matrices(D1, D2, z_thresh=3, plot_type = 'dist'):
    from scipy.stats import zscore
    iu = np.triu_indices_from(D1, k=1)
    d1_flat = D1[iu]
    d2_flat = D2[iu]

    valid = ~np.isnan(d1_flat) & ~np.isnan(d2_flat)
    d1_flat = d1_flat[valid]
    d2_flat = d2_flat[valid]
    i_filtered = iu[0][valid]
    j_filtered = iu[1][valid]

    z_diff = zscore(d2_flat - d1_flat)
    sig_mask = np.abs(z_diff) > z_thresh
    sig_pairs = list(zip(i_filtered[sig_mask], j_filtered[sig_mask]))

    plt.figure(figsize=(6,6))
    plt.scatter(d1_flat[~sig_mask], d2_flat[~sig_mask], s=2, alpha=0.5, label='All pairs', color='cornflowerblue')
    plt.scatter(d1_flat[sig_mask], d2_flat[sig_mask], s=4, alpha=0.7, label='Significant diff', color='yellowgreen')
    # plt.plot([0, max(d1_flat.max(), d2_flat.max())], [0, max(d1_flat.max(), d2_flat.max())], '--', color='gray')
    if plot_type == 'dist':
        plot_text = 'pairwise spatial distance (um)'
    else:
        plot_text = 'log contact'
    plt.xlabel(f'chr7: {plot_text}')
    plt.ylabel(f'ecDNA: {plot_text}')
    corr = np.corrcoef(d1_flat, d2_flat)[0, 1]
    plt.title(f'Correlation = {corr:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return d1_flat, d2_flat, sig_mask, sig_pairs

def plot_sig_pairs(sig_pairs, N=252, title='Significantly Different Bin Pairs'):
    """
    Plots a scatter of significant (i, j) bin pairs, with tick at x=72 and y=72, no lines.

    Parameters:
    - sig_pairs: list of (i, j) tuples
    - N: total number of bins (default 252)
    - title: title of the plot
    """
    import matplotlib.pyplot as plt

    i_sig, j_sig = zip(*sig_pairs) if sig_pairs else ([], [])

    plt.figure(figsize=(6, 6))
    plt.scatter(i_sig, j_sig, s=4, alpha=0.6, color='yellowgreen')
    plt.xlabel('Bin i (linear)')
    plt.ylabel('Bin i (circular)')
    plt.title(title)
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal')

    # Add tick at 72 (no lines)
    xticks = list(plt.xticks()[0])
    yticks = list(plt.yticks()[0])
    if 72 not in xticks:
        xticks.append(72)
    if 72 not in yticks:
        yticks.append(72)
    plt.xticks(sorted(xticks))
    plt.yticks(sorted(yticks))
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.tight_layout()
    plt.show()

    import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

def plot_hic_vs_spatial(hic_mat, spatial_mat, contact_thresh=200, vmax_dist=120, plot_type = 'dist', show_high_low = True):
    """
    Plot Hi-C interaction frequency vs spatial distance with genomic distance as color.
    The scatter plot area is enforced to be square.
    
    Parameters:
    - hic_mat: NxN Hi-C interaction matrix
    - spatial_mat: NxN median spatial distance matrix
    - contact_thresh: threshold to separate high and low contact regions
    - vmax_dist: maximum genomic distance to clip for coloring
    """
    assert hic_mat.shape == spatial_mat.shape, "Shape mismatch between Hi-C and spatial matrices"
    N = hic_mat.shape[0]

    # Get upper triangle indices, excluding diagonal
    iu, ju = np.triu_indices(N, k=1)

    # Extract values
    hic_vals = hic_mat[iu, ju]
    spatial_vals = spatial_mat[iu, ju]
    genomic_dists = np.abs(iu - ju)

    # Filter finite and positive values
    valid = (hic_vals > 0) & np.isfinite(hic_vals) & np.isfinite(spatial_vals)
    hic_vals = hic_vals[valid]
    spatial_vals = spatial_vals[valid]
    genomic_dists = genomic_dists[valid]

    # Compute correlations for high and low contact regions
    is_high = hic_vals > contact_thresh
    is_low = ~is_high

    corr_info = {}
    for label, mask in zip(['Low', 'High'], [is_low, is_high]):
        spearman = spearmanr(hic_vals[mask], spatial_vals[mask])[0]
        pearson = pearsonr(np.log10(hic_vals[mask]), np.log10(spatial_vals[mask]))[0]
        corr_info[label] = (spearman, pearson)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(hic_vals, spatial_vals, c=np.clip(genomic_dists, 0, vmax_dist),
                    cmap='coolwarm', s=5, alpha=0.8, norm=plt.Normalize(0, vmax_dist))

    ax.set_xscale('log')
    ax.set_yscale('log')
    if plot_type == 'dist':
        plot_text = 'pairwise spatial distance (um)'
    else:
        plot_text = 'contact freq(contact if <250nm)'
    
    overall_perason = pearsonr(np.log10(hic_vals), np.log10(spatial_vals))[0]
    ax.set_xlabel('Droplet Hi-C \n pseudobulk interaction frequency')
    ax.set_ylabel(f'multimodal MERFISH \n {plot_text}')
    ax.set_title(f'Hi-C correlation\n Pearson = {overall_perason:.3f}')

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Genomic Distance')

    # Correlation annotations in lower-left corner
    y0 = 0.05
    dy = 0.07
    if show_high_low:
        for i, (label, (s, p)) in enumerate(corr_info.items()):
            ax.text(0.05, y0 + i * dy,
                    f"{label} contact region - Spearman: {s:.2f}, Pearson: {p:.2f}",
                    transform=ax.transAxes, fontsize=9, verticalalignment='bottom')

        # Threshold line
        ax.axvline(contact_thresh, linestyle='--', color='k', linewidth=1)

    # Make the axes box square
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_cross_sample_correlation_subpanel(tracesA, tracesB, name_A, name_B, plot_type, subsample=False):
    ncols = 3
    nrows = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    if subsample:
        tracesB = tracesB[np.random.choice(range(len(tracesB)), size=len(tracesA), replace=False), :, :]

    # --- Panel 1: tracesA ---
    ax = axes[0]
    if plot_type == 'dist':
        matA = np.nanmedian(compute_med_dist_mat(tracesA, contact_dist=None), axis=0)
        im = ax.imshow(matA, cmap='seismic_r', interpolation='nearest', vmin=0.6, vmax=1.2)
    else:
        matA = np.log(compute_med_dist_mat(tracesA, contact_dist=0.25))
        im = ax.imshow(matA, cmap='seismic', interpolation='nearest', vmax=-2)

    ax.set_title(f'{name_A} \n {len(np.unique(tracesA[:, :, -1]))} cells, {len(tracesA)} traces')
    fig.colorbar(im, ax=ax, shrink=0.6)

    # --- Panel 2: tracesB ---
    ax = axes[1]
    if plot_type == 'dist':
        matB = np.nanmedian(compute_med_dist_mat(tracesB, contact_dist=None), axis=0)
        im = ax.imshow(matB, cmap='seismic_r', interpolation='nearest', vmin=0.6, vmax=1.2)
    else:
        matB = np.log(compute_med_dist_mat(tracesB, contact_dist=0.25))
        im = ax.imshow(matB, cmap='seismic', interpolation='nearest', vmax=-2)

    ax.set_title(f'{name_B} \n {len(np.unique(tracesB[:, :, -1]))} cells, {len(tracesB)} traces')
    fig.colorbar(im, ax=ax, shrink=0.6)

    # --- Panel 3: scatter plot + correlation ---
    ax = axes[2]
    A_flat = matA.flatten()
    B_flat = matB.flatten()
    valid = ~(np.isnan(A_flat) | np.isnan(B_flat))

    r, _ = pearsonr(A_flat[valid], B_flat[valid])
    ax.scatter(A_flat[valid], B_flat[valid], alpha=0.5, s=5)
    ax.set_xlabel(name_A)
    ax.set_ylabel(name_B)
    ax.set_title(f'Scatter of pairwise {plot_type} \nPearson r = {r:.2f}')

    fig.suptitle(f'Cross-sample {plot_type} comparison: {name_A} vs {name_B}', fontsize=16)
    plt.tight_layout()
    plt.show()
    

###############################################################
#### Functions import from `Tracer_chrom_copy.ipynb` 091025
###############################################################
def gather_distance_matrix(trace_fls, tr_type = 'Xtraces', nregs = 252,compute_mat=True):
    '''
    traces_all: N_traces x 252 x 6 ## z_um,x_um,y_um, logH,col, cell_id
    
    mats_all: N_traces x 252 x 252
    '''
    dic_header = {'z_um': 0, 'x_um': 1, 'y_um': 2, 'logH': 3, 'col': 4, 'R': 5, 'trace_id': 6}
    mats_all = []
    traces_all = []
    for fl in tqdm(trace_fls):
        Xtr =np.load(fl, allow_pickle = True)[tr_type]
        if len(Xtr) == 0:
            continue
        Xtr = np.concatenate(Xtr)
        cell_id = float(os.path.basename(fl).split('-th')[0].replace('cell',''))
        Xtr = np.concatenate([Xtr,[[cell_id]]*len(Xtr)],axis=-1)
        trace_id = Xtr[:,dic_header['trace_id']].astype(np.uint64)
        H_id = Xtr[:, dic_header['R']].astype(np.int64)
        ntraces = int(np.max(trace_id))+1

        traces = np.zeros([ntraces,nregs,6])+np.nan
        #trace_id
        traces[trace_id,H_id]=Xtr[:,[0,1,2,3,4, -1]] ## z_um,x_um,y_um, logH,col, cell_id
        traces_all.extend(traces) 
        if compute_mat:
            from scipy.spatial.distance import pdist,squareform
            mats = [squareform(pdist(tr))for tr in traces[:,:,:3]]
            mats_all+=mats
    traces_all = np.array(traces_all)
    mats_all = np.array(mats_all)
    return traces_all, mats_all

import numpy as np
from tqdm import tqdm

def filter_overlapping_traces_per_cell(traces_tracer, th_overlap=0.0001, min_cell_size=5500, verbose=True):
    """
    Filters overlapping traces per cell from a 3D array of traces.

    Parameters:
    - traces_tracer: array of shape (N, 252, D), last column should contain integer cell IDs
    - th_overlap: threshold above which traces are considered overlapping
    - min_cell_size: skip cells with more than this many traces (assumed bad or junk)
    - verbose: if True, prints how many traces are kept per cell

    Returns:
    - traces_f: list of arrays of filtered traces per cell
    """

    cell_id = np.nanmean(traces_tracer[:, :, -1], axis=1).astype(int)
    ucells = np.unique(cell_id)
    traces_f = []

    def length_trace(trace):
        return np.sum(~np.isnan(trace[:, 0]))

    for cell in tqdm(ucells, desc='Filtering cells'):
        traces_cell = traces_tracer[cell_id == cell]
        if len(traces_cell) < min_cell_size:
            dic_int = {}
            for i1 in np.arange(len(traces_cell)):
                for i2 in np.arange(i1):
                    tr1, tr2 = traces_cell[i1], traces_cell[i2]
                    valid_mask = ~(np.isnan(tr1[:, 0]) & np.isnan(tr2[:, 0]))
                    if np.sum(valid_mask) > 0:
                        fr = np.sum(np.all(tr1 == tr2, axis=-1)) / np.sum(valid_mask)
                        dic_int[(i1, i2)] = fr

            ints = np.array(list(dic_int.values()))
            if len(ints):
                remove = np.unique([
                    i1 if length_trace(traces_cell[i1]) < length_trace(traces_cell[i2]) else i2
                    for i1, i2 in dic_int
                    if dic_int[(i1, i2)] > th_overlap
                ])
                keep_traces = list(np.setdiff1d(np.arange(len(traces_cell)), remove))

                # Reconsider removed traces if they are low-overlap with all kept traces
                for r in remove:
                    all_cur_traces_int = []
                    for ik in keep_traces:
                        i1, i2 = max(r, ik), min(r, ik)
                        all_cur_traces_int.append(dic_int.get((i1, i2), 0))
                    if np.all(np.array(all_cur_traces_int) < th_overlap):
                        keep_traces.append(r)

                if verbose:
                    print(f'Cell {cell}: {len(keep_traces)} traces kept out of {len(traces_cell)}')

                traces_f.append(traces_cell[keep_traces])
                
    traces_ff = np.concatenate(traces_f)
    return traces_ff


###############################################################
#### Functions import from `Tracer_chrom_copy.ipynb` 091125
#### This section is about deciding chrom copy from anchor points
###############################################################
import numpy as np
from scipy.optimize import linear_sum_assignment

# ------------------ your scoring function (vectorized) ------------------
def logw(x, gd=1, s1=0.085):
    """
    x  : distance in microns (np.ndarray or float)
    gd : genomic distance in units of 5 kb (np.ndarray or float)
    s1 : model parameter (default 0.085)
    Returns: log-weight (same shape as x and gd after broadcasting)
    """
    sigmasq = 0.025**2
    k = (s1*s1 - 2*sigmasq)  # scalar
    ssq = 2*sigmasq + k*gd   # can be vector
    xsq = x*x
    # log of 4*pi*x^2 / (2*pi*ssq)^(3/2)  -  x^2/(2*ssq)
    # guard against non-positive ssq due to pathological params
    ssq = np.maximum(ssq, 1e-12)
    return np.log(4*np.pi*xsq) - 1.5*np.log(2*np.pi*ssq) - xsq/(2*ssq)

# ------------------ genomic axis helpers ------------------
def _build_kb_axis():
    """
    Build cumulative genomic positions (kb) for indices 1..254 (inclusive)
    with special gaps:
      default neighbor gap = 5 kb
      73↔74 = 27 kb
      252↔253 = 50 kb
      253↔254 = 50 kb
    Returns:
      pos_kb: np.ndarray length 255; pos_kb[i] is kb position of locus i (i in 0..254; index 0 unused)
    """
    max_idx = 254
    gaps = np.full(max_idx + 1, 5.0)   # gaps[i] is gap between i and i+1
    gaps[73]  = 27.0
    gaps[252] = 50.0
    gaps[253] = 50.0
    pos_kb = np.zeros(max_idx + 1, dtype=float)
    for i in range(1, max_idx):
        pos_kb[i+1] = pos_kb[i] + gaps[i]
    return pos_kb

# ------------------ data extraction helpers ------------------
def _extract_trace_xyz_and_indices(Xtraces):
    """
    Xtraces: (M, 252, 6) with cols [z,x,y,logH,col,cell_id] in microns.
    Returns:
      xyz_list : list length M; each is (Lt,3) array of valid xyz per trace
      idx_list : list length M; each is (Lt,) array of genomic indices (1..252)
      n_valid  : np.ndarray length M; counts of valid loci per trace
    """
    M, N, _ = Xtraces.shape
    xyz_list, idx_list, n_valid = [], [], np.zeros(M, dtype=int)
    for t in range(M):
        coords = Xtraces[t, :, 0:3]         # z,x,y
        valid = np.all(np.isfinite(coords), axis=1)
        xyz = coords[valid]
        idx = (np.arange(1, N+1))[valid]
        xyz_list.append(xyz)
        idx_list.append(idx)
        n_valid[t] = xyz.shape[0]
    return xyz_list, idx_list, n_valid

def _anchors_from_matrix(A_mat, pix_size, cell_id=None):
    """
    A_mat: (Na, >=3) with z,x,y in cols 0..2; last col MAY be cell_id (optional).
    pix_size: scalar or length-3 (z,x,y) to convert px -> microns for anchors.
    cell_id: if provided, keep only rows where last column == cell_id.
    Returns:
      A_xyz_um: (Na_used, 3) anchors in microns
      sel_rows: indices into A_mat of the kept anchors
    """
    if A_mat is None or A_mat.size == 0:
        return np.empty((0,3)), np.array([], dtype=int)

    if A_mat.ndim != 2 or A_mat.shape[1] < 3:
        raise ValueError("Each anchors matrix must be (N, >=3) with z,x,y in columns 0..2.")

    mask = np.ones(A_mat.shape[0], dtype=bool)
    if cell_id is not None:
        mask &= (A_mat[:, -1] == cell_id)

    sel = np.where(mask)[0]
    if sel.size == 0:
        return np.empty((0,3)), sel

    xyz_pix = A_mat[sel, 0:3]
    ps = np.array(pix_size).reshape(-1)
    if ps.size == 1:
        A_xyz_um = xyz_pix * float(ps[0])
    elif ps.size == 3:
        A_xyz_um = xyz_pix * ps[None, :]
    else:
        raise ValueError("pix_size should be a scalar or a length-3 iterable (z,x,y).")
    return A_xyz_um, sel

# ------------------ core: mean log-score + uniqueness per set ------------------
def _assign_one_set_meanlog(
    set_id,                  # 73 or 253 or 254
    A_xyz_um,                # (Na,3) anchors (µm) for this set
    rows_sel,                # (Na,) indices in original anchors matrix
    Xtraces,                 # (M,252,6)
    xyz_list, idx_list, n_valid,
    pos_kb,                  # genomic kb positions
    Rmax_um=None,            # average distance gate; None => no gate
    score_fn=logw            # callable: (d_um, gd_units) -> log-weight
):
    """
    For one set, compute:
      - mean_logS: (Na, M) mean log-score over all non-NaN loci per trace
      - mean_D:    (Na, M) mean spatial distance over loci per trace
      - matches:   list of (a_idx, t_idx) after Hungarian (uniqueness) and avg-distance pruning
      - used_traces: sorted unique trace indices from matches
    """
    M = Xtraces.shape[0]
    Na = A_xyz_um.shape[0]

    # Precompute gd_units (Lt,) per trace for this set's anchor index
    anchor_kb = pos_kb[int(set_id)]
    gd_units_per_trace = []
    for t in range(M):
        if n_valid[t] == 0:
            gd_units_per_trace.append(np.empty((0,), dtype=float))
            continue
        kb_t = pos_kb[idx_list[t]]
        gd_kb = np.abs(kb_t - anchor_kb)      # (Lt,)
        gd_units_per_trace.append(gd_kb / 5.0)

    # Allocate matrices
    mean_logS = np.full((Na, M), -np.inf, dtype=float)
    mean_D    = np.full((Na, M),  np.inf, dtype=float)

    # Fill matrices
    for a in range(Na):
        a_xyz = A_xyz_um[a]  # (3,)
        for t in range(M):
            Lt = n_valid[t]
            if Lt == 0:
                continue
            # distances to all valid loci of trace t
            diffs = xyz_list[t] - a_xyz    # (Lt,3)
            d_vec = np.linalg.norm(diffs, axis=1)  # (Lt,)
            gd_vec = gd_units_per_trace[t]         # (Lt,)
            # per-locus log-score
            lw = score_fn(d_vec, gd_vec)          # (Lt,)
            # aggregate (all loci)
            mean_logS[a, t] = np.mean(lw)
            mean_D[a, t]    = np.mean(d_vec)

    # Average-distance gate: invalidate pairs with mean_D > Rmax_um
    if Rmax_um is not None:
        bad = mean_D > float(Rmax_um)
        mean_logS[bad] = -np.inf

    # Hungarian: maximize mean_logS  <=>  minimize cost = -mean_logS
    big = 1e12
    cost = np.where(np.isfinite(mean_logS), -mean_logS, big)
    if Na == 0 or M == 0:
        matches, used_traces = [], []
    else:
        rows, cols = linear_sum_assignment(cost)
        matches = []
        for a_idx, t_idx in zip(rows, cols):
            # accept only if finite and (optionally) passes Rmax gate
            if not np.isfinite(mean_logS[a_idx, t_idx]):
                continue
            if Rmax_um is not None and not np.isfinite(mean_D[a_idx, t_idx]):
                continue
            if Rmax_um is not None and mean_D[a_idx, t_idx] > float(Rmax_um):
                continue
            matches.append((int(a_idx), int(t_idx)))

        used_traces = sorted({t for (_, t) in matches})

    result = {
        "anchor_rows": rows_sel,
        "A_xyz_um": A_xyz_um,
        "mean_logS": mean_logS,
        "mean_D": mean_D,
        "matches": matches,
        "used_trace_indices": np.array(used_traces, dtype=int)
    }
    return result

# ------------------ PUBLIC API #1: main assignment (logw-based) ------------------
def assign_traces_logw_per_cell(
    Xtraces,               # (M,252,6) microns; cols: z,x,y,logH,col,cell_id
    anchors_pick,          # list [A73, A253, A254]; each (Na, >=3) in pixels; last col may be cell_id
    pix_size,              # scalar or (z,x,y)
    cell_id=None,          # optional per-cell filter for anchors
    Rmax_um=None,          # average-distance gate (mean distance over loci)
    score_fn=logw          # callable(d_um, gd_units) -> log-weight
):
    """
    Per set (73, 253, 254), compute mean log-scores over all non-NaN loci,
    enforce one-to-one (anchor ↔ trace) via Hungarian (max total score),
    prune by average-distance gate (mean(d) ≤ Rmax_um),
    and return the per-set trace selections + a rich info dict.

    Returns:
      per_set_arrays: dict {
          'Xtraces_chrom_73'  : (M_sub,252,6),
          'Xtraces_chrom_253' : (M_sub,252,6),
          'Xtraces_chrom_254' : (M_sub,252,6),
      }
      info: dict with per-set fields including mean_logS, mean_D, matches, used_trace_indices.
    """
    if Xtraces.ndim != 3 or Xtraces.shape[1] != 252 or Xtraces.shape[2] < 3:
        raise ValueError("Xtraces must be (M,252,>=3) with [z,x,y,...] in microns.")
    if not isinstance(anchors_pick, (list, tuple)) or len(anchors_pick) != 3:
        raise ValueError("anchors_pick must be [X73, X253, X254].")

    # Extract per-trace loci
    xyz_list, idx_list, n_valid = _extract_trace_xyz_and_indices(Xtraces)
    M = Xtraces.shape[0]

    # Genomic axis
    pos_kb = _build_kb_axis()

    # Anchors per set (px->µm, optional per-cell filter)
    A73_um, rows73   = _anchors_from_matrix(anchors_pick[0], pix_size, cell_id=cell_id)
    A253_um, rows253 = _anchors_from_matrix(anchors_pick[1], pix_size, cell_id=cell_id)
    A254_um, rows254 = _anchors_from_matrix(anchors_pick[2], pix_size, cell_id=cell_id)

    # Per-set assignments using mean log-score + Hungarian
    res73  = _assign_one_set_meanlog(73,  A73_um,  rows73,  Xtraces, xyz_list, idx_list, n_valid, pos_kb, Rmax_um, score_fn)
    res253 = _assign_one_set_meanlog(253, A253_um, rows253, Xtraces, xyz_list, idx_list, n_valid, pos_kb, Rmax_um, score_fn)
    res254 = _assign_one_set_meanlog(254, A254_um, rows254, Xtraces, xyz_list, idx_list, n_valid, pos_kb, Rmax_um, score_fn)

    # Slice per-set arrays
    used73  = res73["used_trace_indices"]
    used253 = res253["used_trace_indices"]
    used254 = res254["used_trace_indices"]

    Xtraces_chrom_73  = Xtraces[used73,  :, :] if used73.size  else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    Xtraces_chrom_253 = Xtraces[used253, :, :] if used253.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    Xtraces_chrom_254 = Xtraces[used254, :, :] if used254.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    per_set_arrays = {
        "Xtraces_chrom_73":  Xtraces_chrom_73,
        "Xtraces_chrom_253": Xtraces_chrom_253,
        "Xtraces_chrom_254": Xtraces_chrom_254,
    }

    info = {
        "per_set": {
            73:  res73,
            253: res253,
            254: res254,
        },
        # (Optional) easy access to matrix shapes:
        "meta": {
            "M_traces": int(M),
            "Rmax_um": None if Rmax_um is None else float(Rmax_um),
        }
    }
    return per_set_arrays, info

# ------------------ PUBLIC API #2: intersections + optional save ------------------
def build_trace_intersections_and_save(
    Xtraces, info, save_npz_path=None
):
    """
    From the 'info' produced by assign_traces_logw_per_cell, build:
      - singles:  Xtraces_chrom_73, Xtraces_chrom_253, Xtraces_chrom_254
      - pairs:    Xtraces_chrom_inter_73_253, Xtraces_chrom_inter_73_254, Xtraces_chrom_inter_253_254
      - triple:   Xtraces_chrom_inter_73_253_254
      - union:    Xtraces_inter_any_two  (union of all pairwise intersections)

    If save_npz_path is provided, saves all arrays **and** the `info` dict into a single .npz.
    (Loading requires `allow_pickle=True` and `.item()` for the info dict.)
    """
    import numpy as np

    # used indices per set
    idx73  = np.array(sorted(info["per_set"][73]["used_trace_indices"].tolist()), dtype=int)
    idx253 = np.array(sorted(info["per_set"][253]["used_trace_indices"].tolist()), dtype=int)
    idx254 = np.array(sorted(info["per_set"][254]["used_trace_indices"].tolist()), dtype=int)

    # singles
    X_chrom_73  = Xtraces[idx73,  :, :] if idx73.size  else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    X_chrom_253 = Xtraces[idx253, :, :] if idx253.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    X_chrom_254 = Xtraces[idx254, :, :] if idx254.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    # pairwise intersections
    I_73_253  = np.array(sorted(set(idx73)  & set(idx253)), dtype=int)
    I_73_254  = np.array(sorted(set(idx73)  & set(idx254)), dtype=int)
    I_253_254 = np.array(sorted(set(idx253) & set(idx254)), dtype=int)

    X_inter_73_253  = Xtraces[I_73_253,  :, :] if I_73_253.size  else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    X_inter_73_254  = Xtraces[I_73_254,  :, :] if I_73_254.size  else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    X_inter_253_254 = Xtraces[I_253_254, :, :] if I_253_254.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    # triple intersection
    I_73_253_254 = np.array(sorted(set(idx73) & set(idx253) & set(idx254)), dtype=int)
    X_inter_73_253_254 = Xtraces[I_73_253_254, :, :] if I_73_253_254.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    # union of any two
    I_any_two = np.unique(np.concatenate([I_73_253, I_73_254, I_253_254])) if (
        I_73_253.size or I_73_254.size or I_253_254.size
    ) else np.array([], dtype=int)
    X_any_two = Xtraces[I_any_two, :, :] if I_any_two.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    selection_dict = {
        "Xtraces_chrom_73": X_chrom_73,
        "Xtraces_chrom_253": X_chrom_253,
        "Xtraces_chrom_254": X_chrom_254,
        "Xtraces_chrom_inter_73_253": X_inter_73_253,
        "Xtraces_chrom_inter_73_254": X_inter_73_254,
        "Xtraces_chrom_inter_253_254": X_inter_253_254,
        "Xtraces_chrom_inter_73_253_254": X_inter_73_253_254,
        "Xtraces_inter_any_two": X_any_two,
    }
    indices_dict = {
        "Xtraces_chrom_73": idx73,
        "Xtraces_chrom_253": idx253,
        "Xtraces_chrom_254": idx254,
        "Xtraces_chrom_inter_73_253": I_73_253,
        "Xtraces_chrom_inter_73_254": I_73_254,
        "Xtraces_chrom_inter_253_254": I_253_254,
        "Xtraces_chrom_inter_73_253_254": I_73_253_254,
        "Xtraces_inter_any_two": I_any_two,
    }

    if save_npz_path is not None:
        # Save arrays AND info dict together
        to_save = dict(selection_dict)
        to_save["info"] = info  # will be pickled inside the npz
        np.savez_compressed(save_npz_path, **to_save)

    return selection_dict, indices_dict


import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

def build_superpoints_triplets_then_new_pairs_from_unused(
    info: dict,
    thresh_um: float = 2.0,
    triplet_score: str = "sum_pair_dists",   # or "max_pair_dist"
) -> Tuple[List[Dict[int, int]], dict]:
    """
    Policy:
      1) Select disjoint 3-point SPs (73,253,254) first (all three pairwise dists ≤ thresh_um).
      2) Then add 2-point SPs (73-253, 73-254, 253-254) using ONLY anchors that were NOT used
         in any selected triplet. (Edges of triplets are implicitly excluded.)
      3) Pair matching per pair-type uses Hungarian; each anchor used at most once in that type.

    Returns:
      superpoints: [triplets..., new_pairs...]
      qc: counts & basic stats
    """
    per = info["per_set"]
    A73  = per.get(73,  {}).get("A_xyz_um", np.empty((0,3)))
    A253 = per.get(253, {}).get("A_xyz_um", np.empty((0,3)))
    A254 = per.get(254, {}).get("A_xyz_um", np.empty((0,3)))

    N73, N253, N254 = A73.shape[0], A253.shape[0], A254.shape[0]
    superpoints: List[Dict[int, int]] = []

    # ---- helpers ----
    def _pdist(A, B):
        if A.size == 0 or B.size == 0:
            return np.empty((A.shape[0], B.shape[0]))
        diffs = A[:, None, :] - B[None, :, :]
        return np.linalg.norm(diffs, axis=2)

    def _pair_matching_remaining(A_coords, B_coords, thresh, allowA, allowB):
        """
        Hungarian per pair-type on remaining (allow*) anchors only.
        Returns list of matched pairs (iA, iB). Each node used at most once.
        """
        idxA = np.where(allowA)[0]
        idxB = np.where(allowB)[0]
        if idxA.size == 0 or idxB.size == 0:
            return []

        D = _pdist(A_coords[idxA], B_coords[idxB])
        if D.size == 0:
            return []

        big = 1e6
        cost = D.copy()
        cost[D > thresh] = big

        # pad to square for Hungarian
        nA, nB = cost.shape
        if nA > nB:
            pad = np.full((nA, nA - nB), big)
            cost_mat = np.hstack([cost, pad])
            cols_are_real = np.array([True]*nB + [False]*(nA-nB))
        elif nB > nA:
            pad = np.full((nB - nA, nB), big)
            cost_mat = np.vstack([cost, pad])
            cols_are_real = np.array([True]*nB)
        else:
            cost_mat = cost
            cols_are_real = np.array([True]*cost.shape[1])

        rows, cols = linear_sum_assignment(cost_mat)
        matches = []
        for r, c in zip(rows, cols):
            if r < nA and c < cost.shape[1] and cols_are_real[c]:
                if cost[r, c] < big:
                    iA = int(idxA[r])
                    iB = int(idxB[c])
                    matches.append((iA, iB))
        return matches

    # ---- distances & adjacency ----
    D_73_253  = _pdist(A73,  A253)  if (N73 and N253) else np.empty((N73, N253))
    D_73_254  = _pdist(A73,  A254)  if (N73 and N254) else np.empty((N73, N254))
    D_253_254 = _pdist(A253, A254)  if (N253 and N254) else np.empty((N253, N254))

    A_73_253  = (D_73_253  <= thresh_um) if D_73_253.size  else np.zeros((N73, N253),  dtype=bool)
    A_73_254  = (D_73_254  <= thresh_um) if D_73_254.size  else np.zeros((N73, N254),  dtype=bool)
    A_253_254 = (D_253_254 <= thresh_um) if D_253_254.size else np.zeros((N253, N254), dtype=bool)

    # ---- 1) enumerate triplet candidates & select disjoint set ----
    triplet_candidates = []  # (score, i, j, k)
    if N73 and N253 and N254:
        for i in range(N73):
            js = np.where(A_73_253[i, :])[0]
            ks = np.where(A_73_254[i, :])[0]
            if js.size == 0 or ks.size == 0:
                continue
            for j in js:
                ks2 = ks[A_253_254[j, ks]]
                for k in ks2:
                    d1 = D_73_253[i, j]
                    d2 = D_73_254[i, k]
                    d3 = D_253_254[j, k]
                    score = max(d1, d2, d3) if triplet_score == "max_pair_dist" else (d1 + d2 + d3)
                    triplet_candidates.append((score, i, j, k))

    triplet_candidates.sort(key=lambda x: x[0])
    used73  = np.zeros(N73,  dtype=bool)
    used253 = np.zeros(N253, dtype=bool)
    used254 = np.zeros(N254, dtype=bool)

    selected_triplets = []
    for score, i, j, k in triplet_candidates:
        if used73[i] or used253[j] or used254[k]:
            continue
        selected_triplets.append((i, j, k))
        used73[i] = used253[j] = used254[k] = True

    # append triplets first
    for (i, j, k) in selected_triplets:
        superpoints.append({73: i, 253: j, 254: k})

    # ---- 2) add NEW pairs ONLY from anchors not used in any triplet ----
    # Remaining-allowed masks per set:
    allow73  = ~used73
    allow253 = ~used253
    allow254 = ~used254

    pairs_new_counts = {"73_253": 0, "73_254": 0, "253_254": 0}

    # 73-253
    if N73 and N253:
        matches = _pair_matching_remaining(A73, A253, thresh_um, allow73, allow253)
        # use-once per type (the matcher already enforces that on the remaining subset)
        for i, j in matches:
            if A_73_253[i, j] and allow73[i] and allow253[j]:
                superpoints.append({73: i, 253: j})
                allow73[i] = False
                allow253[j] = False
                pairs_new_counts["73_253"] += 1

    # 73-254
    if N73 and N254:
        matches = _pair_matching_remaining(A73, A254, thresh_um, allow73, allow254)
        for i, k in matches:
            if A_73_254[i, k] and allow73[i] and allow254[k]:
                superpoints.append({73: i, 254: k})
                allow73[i] = False
                allow254[k] = False
                pairs_new_counts["73_254"] += 1

    # 253-254
    if N253 and N254:
        matches = _pair_matching_remaining(A253, A254, thresh_um, allow253, allow254)
        for j, k in matches:
            if A_253_254[j, k] and allow253[j] and allow254[k]:
                superpoints.append({253: j, 254: k})
                allow253[j] = False
                allow254[k] = False
                pairs_new_counts["253_254"] += 1

    qc = {
        "thresh_um": float(thresh_um),
        "n73": N73, "n253": N253, "n254": N254,
        "n_triplets_candidate": len(triplet_candidates),
        "n_triplets_selected": len(selected_triplets),
        "n_pairs_new": pairs_new_counts,
    }
    return superpoints, qc
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional, List, Dict

def _ensure_minD(info: dict, Xtraces: np.ndarray) -> dict:
    """
    Ensure info['per_set'][sid]['min_D'] exists for sid in {73,253,254}.
    min_D[a, t] = min over loci L of || anchor_a - trace_t[L] ||_2 (µm), ignoring NaNs.
    """
    per = info["per_set"]
    # traces: (M, 252, 6) with columns [z, x, y, logH, col, cell_id] in µm
    Z = Xtraces[:, :, 0]; X = Xtraces[:, :, 1]; Y = Xtraces[:, :, 2]
    M, L = Z.shape
    T = np.stack([Z, X, Y], axis=-1)  # (M, L, 3)

    for sid in (73, 253, 254):
        ps = per.get(sid, {})
        A = ps.get("A_xyz_um", None)   # (Na, 3)
        if A is None or A.size == 0:
            per.setdefault(sid, {})["min_D"] = np.empty((0, M))
            continue
        if "min_D" in ps and ps["min_D"].shape == (A.shape[0], M):
            continue

        Na = A.shape[0]
        minD = np.full((Na, M), np.inf, dtype=float)
        for a_idx in range(Na):
            a = A[a_idx]                # (3,)
            dists = np.linalg.norm(T - a, axis=-1)  # (M, L)
            with np.errstate(invalid="ignore"):
                md = np.nanmin(dists, axis=1)       # (M,)
            md[~np.isfinite(md)] = np.inf
            minD[a_idx, :] = md
        per[sid]["min_D"] = minD
    info["per_set"] = per
    return info

def build_trace_intersections_and_save_joint(
    Xtraces: np.ndarray,
    info: dict,
    superpoints: List[Dict[int, int]],
    save_npz_path: Optional[str] = None,
    # --- combined gating knobs ---
    Rmax_min_um: Optional[float] = None,     # gate on min_D ≤ Rmax_min_um
    Rmax_mean_um: Optional[float] = None,    # gate on mean_D ≤ Rmax_mean_um
    gate_logic: str = "and",                 # "and" (default) or "or"
    # --- legacy/back-compat (ignored if either Rmax_* provided) ---
    gate_mode: str = "mean",                 # "mean" or "min"; used only if both Rmax_* are None
    topk_frac: float = 0.1,                  # reserved for future "topk_mean"
    # --- assignment knob ---
    unassign_below: Optional[float] = None   # if set, ASPs with joint S < threshold can go to dummy
):
    """
    Joint (ASP-based) intersections with combined distance gating and penalized dummy columns.

    Saves:
      - Xtraces_chrom_73 / _253 / _254 (singles via used_trace_indices you provided)
      - Intersections: _73_253, _73_254, _253_254, _73_253_254, and Xtraces_inter_any_two
      - 'info' and 'info_joint'
    """
    # ---------- resolve thresholds ----------
    meta_R = info.get("meta", {}).get("Rmax_um", None)
    if Rmax_min_um is None and Rmax_mean_um is None and meta_R is not None:
        if (gate_mode or "").lower() == "min":
            Rmax_min_um = float(meta_R)
        else:
            Rmax_mean_um = float(meta_R)

    use_min_gate  = (Rmax_min_um  is not None)
    use_mean_gate = (Rmax_mean_um is not None)

    if use_min_gate:
        _ensure_minD(info, Xtraces)

    M = Xtraces.shape[0]
    A = len(superpoints)
    per = info["per_set"]
    for sid in (73, 253, 254):
        per.setdefault(sid, {})
        per[sid].setdefault("mean_logS", np.empty((0, M)))
        per[sid].setdefault("mean_D",    np.empty((0, M)))
        per[sid].setdefault("min_D",     np.empty((0, M)))  # may remain empty if not used

    # ---------- gate helper (combined) ----------
    def _passes_gate(set_id: int, anchor_idx: int) -> np.ndarray:
        if not (use_min_gate or use_mean_gate):
            return np.ones(M, dtype=bool)

        mmin  = None
        mmean = None
        if use_min_gate:
            minD = per[set_id]["min_D"]
            mmin = (minD[anchor_idx, :] <= float(Rmax_min_um)) if minD.shape[0] > anchor_idx else np.zeros(M, bool)
        if use_mean_gate:
            meanD = per[set_id]["mean_D"]
            mmean = (meanD[anchor_idx, :] <= float(Rmax_mean_um)) if meanD.shape[0] > anchor_idx else np.zeros(M, bool)

        if use_min_gate and use_mean_gate:
            return (mmin & mmean) if (gate_logic.lower() == "and") else (mmin | mmean)
        elif use_min_gate:
            return mmin
        else:
            return mmean

    # ---------- joint score matrix S (A x M) ----------
    S = np.full((A, M), -np.inf, dtype=float)
    asp_list = [{"sets": tuple(sorted(sp.keys())), "anchors": dict(sp)} for sp in superpoints]

    for a_idx, asp in enumerate(superpoints):
        joint = np.zeros(M, dtype=float)
        feasible = np.ones(M, dtype=bool)
        for sid, anchor_idx in asp.items():
            mS = per[sid]["mean_logS"]
            if anchor_idx < 0 or anchor_idx >= mS.shape[0]:
                feasible[:] = False
                break
            joint += mS[anchor_idx, :]
            feasible &= _passes_gate(sid, anchor_idx)
        S[a_idx, feasible] = joint[feasible]

    # ---------- optional: unassign below threshold ----------
    mask_bad = (S < float(unassign_below)) if (unassign_below is not None) else np.zeros_like(S, dtype=bool)

    # ---------- penalized dummy columns ----------
    big = 1e12
    cost_real = np.where(np.isfinite(S) & (~mask_bad), -S, big)
    finite_costs = cost_real[np.isfinite(cost_real)]
    DUMMY_PENALTY = float(np.max(finite_costs) + 1.0) if finite_costs.size else 1e9
    dummy = np.full((A, A), DUMMY_PENALTY, dtype=float) if A > 0 else np.zeros((0, 0), dtype=float)
    cost = np.hstack([cost_real, dummy]) if A > 0 else cost_real

    # ---------- Hungarian assignment ----------
    matches = []
    if A > 0 and cost.size > 0:
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if c < M and np.isfinite(S[r, c]) and not mask_bad[r, c]:
                matches.append((int(r), int(c)))

    # ---------- bucket results ----------
    combo_keys = {
        frozenset((73, 253)): "Xtraces_chrom_inter_73_253",
        frozenset((73, 254)): "Xtraces_chrom_inter_73_254",
        frozenset((253, 254)): "Xtraces_chrom_inter_253_254",
        frozenset((73, 253, 254)): "Xtraces_chrom_inter_73_253_254",
    }
    combo_to_indices = {k: set() for k in combo_keys.keys()}
    match_info = []

    for asp_idx, t_idx in matches:
        asp = superpoints[asp_idx]
        sets_fs = frozenset(asp.keys())
        if sets_fs in combo_to_indices:
            combo_to_indices[sets_fs].add(t_idx)

        per_set_breakdown = {}
        joint_score = float(S[asp_idx, t_idx])
        for sid, anchor_idx in asp.items():
            per_set_breakdown[int(sid)] = {
                "anchor_idx": int(anchor_idx),
                "mean_logS": float(per[sid]["mean_logS"][anchor_idx, t_idx]) if per[sid]["mean_logS"].size else np.nan,
                "mean_D":    float(per[sid]["mean_D"][anchor_idx, t_idx])    if per[sid]["mean_D"].size    else np.nan,
                "min_D":     float(per[sid]["min_D"][anchor_idx, t_idx])     if per[sid]["min_D"].size     else np.nan,
            }
        match_info.append({
            "asp_index": asp_idx,
            "trace_idx": t_idx,
            "sets": tuple(sorted(asp.keys())),
            "joint_score": joint_score,
            "per_set": per_set_breakdown,
        })

    def _slice(indices_set):
        if not indices_set:
            return np.empty((0, Xtraces.shape[1], Xtraces.shape[2])), np.array([], dtype=int)
        idx = np.array(sorted(indices_set), dtype=int)
        return Xtraces[idx, :, :], idx

    X_73_253, idx_73_253 = _slice(combo_to_indices[frozenset((73, 253))])
    X_73_254, idx_73_254 = _slice(combo_to_indices[frozenset((73, 254))])
    X_253_254, idx_253_254 = _slice(combo_to_indices[frozenset((253, 254))])
    X_73_253_254, idx_73_253_254 = _slice(combo_to_indices[frozenset((73, 253, 254))])

    idx_any_two = np.unique(
        np.concatenate([idx_73_253, idx_73_254, idx_253_254])
    ) if (idx_73_253.size or idx_73_254.size or idx_253_254.size) else np.array([], dtype=int)
    X_any_two = Xtraces[idx_any_two, :, :] if idx_any_two.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    # ---------- singles from per-set used_trace_indices (your exact snippet) ----------
    idx73  = np.array(sorted(info["per_set"][73].get("used_trace_indices", []).tolist()),  dtype=int) \
             if "used_trace_indices" in info["per_set"][73] else np.array([], dtype=int)
    idx253 = np.array(sorted(info["per_set"][253].get("used_trace_indices", []).tolist()), dtype=int) \
             if "used_trace_indices" in info["per_set"][253] else np.array([], dtype=int)
    idx254 = np.array(sorted(info["per_set"][254].get("used_trace_indices", []).tolist()), dtype=int) \
             if "used_trace_indices" in info["per_set"][254] else np.array([], dtype=int)

    X_chrom_73  = Xtraces[idx73,  :, :] if idx73.size  else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    X_chrom_253 = Xtraces[idx253, :, :] if idx253.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))
    X_chrom_254 = Xtraces[idx254, :, :] if idx254.size else np.empty((0, Xtraces.shape[1], Xtraces.shape[2]))

    selection_dict = {
        "Xtraces_chrom_73": X_chrom_73,
        "Xtraces_chrom_253": X_chrom_253,
        "Xtraces_chrom_254": X_chrom_254,
        "Xtraces_chrom_inter_73_253": X_73_253,
        "Xtraces_chrom_inter_73_254": X_73_254,
        "Xtraces_chrom_inter_253_254": X_253_254,
        "Xtraces_chrom_inter_73_253_254": X_73_253_254,
        "Xtraces_inter_any_two": X_any_two,
    }
    indices_dict = {
        "Xtraces_chrom_inter_73_253": idx_73_253,
        "Xtraces_chrom_inter_73_254": idx_73_254,
        "Xtraces_chrom_inter_253_254": idx_253_254,
        "Xtraces_chrom_inter_73_253_254": idx_73_253_254,
        "Xtraces_inter_any_two": idx_any_two,
    }

    info_joint = {
        "Rmax_min_um": Rmax_min_um,
        "Rmax_mean_um": Rmax_mean_um,
        "gate_logic": gate_logic,
        "legacy_gate_mode": gate_mode,
        "topk_frac": topk_frac,
        "unassign_below": unassign_below,
        "asp_list": asp_list,
        "S": S,
        "matches": match_info,
        "used_trace_indices": np.array(sorted({m["trace_idx"] for m in match_info}), dtype=int),
        "by_combo": {
            "73_253": idx_73_253,
            "73_254": idx_73_254,
            "253_254": idx_253_254,
            "73_253_254": idx_73_253_254,
            "any_two": idx_any_two,
        }
    }

    if save_npz_path is not None:
        to_save = dict(selection_dict)
        to_save["info"] = info
        to_save["info_joint"] = info_joint
        np.savez_compressed(save_npz_path, **to_save)

    return selection_dict, indices_dict, info_joint



###############################################################
#### Functions import from `Tracer_intron_colocalization.ipynb` 091125
###############################################################
import numpy as np
from scipy.optimize import linear_sum_assignment

def assign_traces_to_XQ_mean_gate_dual(
    linear_traces,      # (M1,L,6) in µm
    circular_traces,    # (M2,L,6) in µm
    XQ_pixels,          # (N,>=3) in pixels; first 3 cols are z,x,y
    pix_size,           # scalar or (3,) array for (z,x,y) pixel size
    Rmax_um=1.5,
    save_npz_path=None,
):
    """
    Assign anchors (XQ) uniquely to either linear or circular traces
    by minimizing mean 3D distance (over non-NA loci).
    Each anchor -> at most 1 trace. Hard gate: mean_dist <= Rmax_um.

    Returns separate outputs for linear and circular pools.
    """
    # 1) convert anchors to µm
    XQ_pix = np.asarray(XQ_pixels)
    if np.isscalar(pix_size):
        pz = px = py = float(pix_size)
    else:
        pix_size = np.asarray(pix_size).astype(float)
        assert pix_size.shape == (3,)
        pz, px, py = pix_size.tolist()
    XQ_um = np.c_[XQ_pix[:, 0] * pz, XQ_pix[:, 1] * px, XQ_pix[:, 2] * py]

    # 2) concat traces
    M_lin = linear_traces.shape[0]
    M_circ = circular_traces.shape[0]
    if M_lin and M_circ:
        Xtraces_all = np.concatenate([linear_traces, circular_traces], axis=0)
    elif M_lin:
        Xtraces_all = linear_traces
    else:
        Xtraces_all = circular_traces
    M = Xtraces_all.shape[0]
    L = Xtraces_all.shape[1]

    # 3) compute mean distances
    N = XQ_um.shape[0]
    Z = Xtraces_all[:, :, 0]; X = Xtraces_all[:, :, 1]; Y = Xtraces_all[:, :, 2]
    T = np.stack([Z, X, Y], axis=-1)  # (M,L,3)

    D_mean = np.full((N, M), np.inf)
    for a in range(N):
        diffs = T - XQ_um[a]
        dists = np.linalg.norm(diffs, axis=-1)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(dists, axis=1)
        mu[~np.isfinite(mu)] = np.inf
        D_mean[a, :] = mu

    # 4) build cost with Rmax gate
    big = 1e12
    cost_real = D_mean.copy()
    cost_real[D_mean > Rmax_um] = big
    finite_real = cost_real[(cost_real < big)]
    DUMMY_PENALTY = float(np.max(finite_real) + 1e-6) if finite_real.size else Rmax_um + 1e-6
    dummy = np.full((N, N), DUMMY_PENALTY)
    cost = np.hstack([cost_real, dummy])

    # 5) Hungarian assignment
    assignments_linear = []
    assignments_circular = []
    if N > 0 and cost.size > 0:
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if c < M and cost[r, c] < big:
                if c < M_lin:  # linear trace
                    assignments_linear.append({
                        "anchor_index": int(r),
                        "trace_index": int(c),
                        "mean_distance": float(D_mean[r, c]),
                    })
                else:           # circular trace
                    assignments_circular.append({
                        "anchor_index": int(r),
                        "trace_index": int(c - M_lin),
                        "mean_distance": float(D_mean[r, c]),
                    })
            # else: dummy (unassigned)

    # 6) pack outputs
    sel_lin = np.array(sorted({a["trace_index"] for a in assignments_linear}), int)
    sel_circ = np.array(sorted({a["trace_index"] for a in assignments_circular}), int)

    lin_selected = linear_traces[sel_lin, :, :] if sel_lin.size else np.empty((0, L, 6))
    circ_selected = circular_traces[sel_circ, :, :] if sel_circ.size else np.empty((0, L, 6))

    result = {
        "XQ_um": XQ_um,
        "D_mean": D_mean,
        "assignments_linear": assignments_linear,
        "assignments_circular": assignments_circular,
        "selected_trace_indices_linear": sel_lin,
        "selected_trace_indices_circular": sel_circ,
        "linear_traces_selected": lin_selected,
        "circular_traces_selected": circ_selected,
    }

    if save_npz_path is not None:
        np.savez_compressed(save_npz_path, **result)

    return result


import numpy as np

def compute_neighbor_density_matrices_per_cell(linear_traces_cell, circular_traces_cell, N=5):
    """
    Inputs
    ------
    linear_traces_cell   : (M_lin, 252, 6) in µm, cols [z, x, y, logH, col, cell_id]
    circular_traces_cell : (M_circ, 252, 6) in µm, same format
    N                    : number of nearest neighbors to average (uses all available if < N)

    Returns
    -------
    linear_mat   : (M_lin, 6)  columns = [z_cent, x_cent, y_cent, avg_dist_NN, trace_id, cell_id]
    circular_mat : (M_circ, 6) columns = [z_cent, x_cent, y_cent, avg_dist_NN, trace_id, cell_id]

    Notes
    -----
    - Neighbors are drawn from the *combined* pool (linear + circular).
    - If a trace has no other valid centroids, avg_dist_NN = NaN.
    - cell_id is taken from the first non-NaN value in column -1 of the trace.
    """
    linear_traces_cell   = np.asarray(linear_traces_cell)
    circular_traces_cell = np.asarray(circular_traces_cell)

    M_lin  = linear_traces_cell.shape[0]   if linear_traces_cell.size   else 0
    M_circ = circular_traces_cell.shape[0] if circular_traces_cell.size else 0
    M_all  = M_lin + M_circ

    # Early return with correct shapes if empty
    if M_all == 0:
        return np.empty((0, 6), float), np.empty((0, 6), float)

    # ---- helpers ----
    def centroid_zxy(trace):
        z = trace[:, 0]; x = trace[:, 1]; y = trace[:, 2]
        valid = ~(np.isnan(z) | np.isnan(x) | np.isnan(y))
        if not np.any(valid):
            return np.array([np.nan, np.nan, np.nan], float)
        return np.array([np.nanmean(z[valid]), np.nanmean(x[valid]), np.nanmean(y[valid])], float)

    def cell_id_from_trace(trace):
        cid_col = trace[:, -1]
        mask = ~np.isnan(cid_col)
        return int(cid_col[mask][0]) if np.any(mask) else -1

    # ---- centroids & metadata for combined pool ----
    centroids = np.full((M_all, 3), np.nan, float)
    cell_ids  = np.full(M_all, -1, int)
    is_valid  = np.zeros(M_all, bool)

    # linear: global indices [0 .. M_lin-1]
    for i in range(M_lin):
        c = centroid_zxy(linear_traces_cell[i])
        centroids[i] = c
        cell_ids[i]  = cell_id_from_trace(linear_traces_cell[i])
        is_valid[i]  = np.all(np.isfinite(c))

    # circular: global indices [M_lin .. M_lin+M_circ-1]
    for j in range(M_circ):
        gi = M_lin + j
        c = centroid_zxy(circular_traces_cell[j])
        centroids[gi] = c
        cell_ids[gi]  = cell_id_from_trace(circular_traces_cell[j])
        is_valid[gi]  = np.all(np.isfinite(c))

    # ---- pairwise distances among valid centroids (global) ----
    D = np.full((M_all, M_all), np.nan, float)
    valid_idx = np.where(is_valid)[0]
    if valid_idx.size >= 1:
        C = centroids[valid_idx]  # (K,3)
        diffs   = C[:, None, :] - C[None, :, :]
        D_valid = np.linalg.norm(diffs, axis=2)  # (K,K)
        np.fill_diagonal(D_valid, np.nan)
        for ii, gi in enumerate(valid_idx):
            D[gi, valid_idx] = D_valid[ii]

    # ---- average distance to nearest N neighbors (global) ----
    avg_dist = np.full(M_all, np.nan, float)
    for i in range(M_all):
        row = D[i]
        nb = row[np.isfinite(row)]
        if nb.size > 0:
            k = min(N, nb.size)
            # use partial sort for efficiency
            kth = np.partition(nb, k-1)[:k]
            avg_dist[i] = float(np.mean(kth))

    # ---- build outputs (per pool) ----
    # linear matrix
    if M_lin:
        lin_cent  = centroids[:M_lin]
        lin_avg   = avg_dist[:M_lin]
        lin_ids   = np.arange(M_lin, dtype=float)  # trace_id within linear pool
        lin_cids  = cell_ids[:M_lin].astype(float)
        linear_mat = np.column_stack([lin_cent, lin_avg, lin_ids, lin_cids])
    else:
        linear_mat = np.empty((0, 6), float)

    # circular matrix
    if M_circ:
        cir_cent  = centroids[M_lin:M_lin+M_circ]
        cir_avg   = avg_dist[M_lin:M_lin+M_circ]
        cir_ids   = np.arange(M_circ, dtype=float)  # trace_id within circular pool
        cir_cids  = cell_ids[M_lin:M_lin+M_circ].astype(float)
        circular_mat = np.column_stack([cir_cent, cir_avg, cir_ids, cir_cids])
    else:
        circular_mat = np.empty((0, 6), float)

    return linear_mat, circular_mat


import numpy as np

def build_trace_assignment_table_per_cell(
    out,                 # result from assign_traces_to_XQ_mean_gate_dual
    XQ_pixels,           # (N, >=5), brightness in column -5
    linear_NN_mat,       # (M_lin, 6): zc, xc, yc, avgNN, trace_id, cell_id
    circular_NN_mat,     # (M_circ, 6): zc, xc, yc, avgNN, trace_id, cell_id
    brightness_for_unassigned=0.0,  # use 0 for missing anchors (or np.nan if you prefer)
    return_pandas=True
):
    """
    Build a tidy per-trace table linking neighbor density with anchor brightness.
    Returns:
      - (tbl_rec, tbl_df) if return_pandas=True and pandas available
      - (tbl_rec, None)   otherwise

    Columns:
      cell_id (int)
      pool ('linear'|'circular')
      trace_id (int)     # within its pool
      avg_NN_dist (float)
      has_anchor (bool)
      anchor_index (int) # -1 if none
      anchor_brightness (float)
    """
    XQ_pixels = np.asarray(XQ_pixels)
    # brightness from the anchor array:
    anchor_brightness = XQ_pixels[:, -5] if XQ_pixels.shape[1] >= 5 else np.array([])

    # maps: trace -> anchor for each pool
    lin_map  = {a["trace_index"]: a["anchor_index"] for a in out.get("assignments_linear", [])}
    circ_map = {a["trace_index"]: a["anchor_index"] for a in out.get("assignments_circular", [])}

    # helper to build rows from a pool
    rows = []

    def _ingest_pool(NN_mat, pool_name, amap):
        if NN_mat.size == 0:
            return
        # columns: [zc, xc, yc, avgNN, trace_id, cell_id]
        avgNN   = NN_mat[:, 3].astype(float)
        traceid = NN_mat[:, 4].astype(int)
        cellid  = NN_mat[:, 5].astype(int)
        for ti, d, cid in zip(traceid, avgNN, cellid):
            if ti in amap:
                ai = amap[ti]
                b  = float(anchor_brightness[ai]) if anchor_brightness.size and 0 <= ai < anchor_brightness.size else float(brightness_for_unassigned)
                rows.append((cid, pool_name, ti, float(d), True,  int(ai), b))
            else:
                rows.append((cid, pool_name, ti, float(d), False, -1, float(brightness_for_unassigned)))

    _ingest_pool(linear_NN_mat,   "linear",   lin_map)
    _ingest_pool(circular_NN_mat, "circular", circ_map)

    dtype = np.dtype([
        ('cell_id',         'i4'),
        ('pool',            'U9'),   # 'linear' / 'circular'
        ('trace_id',        'i4'),
        ('avg_NN_dist',     'f8'),
        ('has_anchor',      'b1'),
        ('anchor_index',    'i4'),
        ('anchor_brightness','f8'),
    ])
    tbl_rec = np.array(rows, dtype=dtype)

    if return_pandas:
        try:
            import pandas as pd
            tbl_df = pd.DataFrame.from_records(tbl_rec)
            return tbl_rec, tbl_df
        except Exception:
            pass
    return tbl_rec, None
