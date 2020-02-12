import numpy as np
import os
from glob import glob
from skimage import io
import scipy
import scipy.ndimage.morphology as morp
from skimage.morphology import skeletonize_3d
from itertools import combinations
import copy as cp

def centroid(y, x):
    """Estimate centroid coordinates"""
    length = float(len(x))
    return round(np.sum(y)/length, 0), round(np.sum(x)/length, 0)

def indx_nearest_neighbours(point,arr, n):
    """Return the index of the closest element in arr"""
    tree=scipy.spatial.cKDTree(arr)
    dist,ind=tree.query(point, k=n)
    return ind, dist

def import_image_stack(path, n):
    """Import tiff image sequence
    path = path to the stack of images
    n = plugin to use"""
    os.chdir(path)
    lst = sorted (glob("*.tif"))
    modules = ['imageio','pil','tifffile','matplotlib']
    module=[]
    for n, i in enumerate(modules):
        if np.array(io.imread(lst[0], plugin=i)).ndim==2:
            module = modules[n]
            break
    if not module:
        raise ValueError("Cannot open mask properly with skimage.io")
    img = [io.imread (i, plugin=module) for i in lst]
    img = np.array(img, dtype='uint8')
    print ("Image shape is:\t{}\n".format(img.shape))
    return img

def skeletonize (img):
    """Skeletonize image"""
    print ("\nSkeletonizing image...")
    s = np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]])
    skl = []
    for i in img:
        j = skeletonize_3d(i)
        k = morp.binary_hit_or_miss(j, structure1=s)
        k_ind = k.nonzero()
        k_ind = np.column_stack(k_ind)
        if len(k_ind)>0:
            for el in k_ind:
                j[el[0], el[1]]=0
        skl.append(j)
    skl = np.array(skl, dtype=int, copy=False)
    skl[skl>0]=1
    print ("Image skeletonized.\n")
    return skl

def find_interconnections(skl):
    """It finds aand removes interconnection in the skeletonized image"""
    print ("\nRemoving interconnections...")
    vx = 3
    eg = vx//2
    skl_copy = np.array(skl.copy()).astype(int)
    skl_copy = np.pad(skl_copy, ((eg,eg), (eg,eg), (eg,eg)), mode='constant', constant_values=0)
    skl_inv = abs(skl_copy-1)
    z, y, x  = np.nonzero(skl_copy)
    nnz_ind = np.column_stack([z, y, x])
    int_3 = []
    for i in nnz_ind:
        arr2d = skl_inv[i[0], i[1]-eg:i[1]+vx-eg, i[2]-eg:i[2]+vx-eg]
        _, n = scipy.ndimage.measurements.label(arr2d)
        if n>2:
            int_3.append([i[0],i[1],i[2]])
            skl_copy[i[0], i[1]-eg:i[1]+vx-eg, i[2]-eg:i[2]+vx-eg]=0
    del skl_inv
    int_3 = np.array(int_3, copy=False)-eg
    skl_int = skl_copy[eg:-eg,eg:-eg,eg:-eg]
    del skl_copy
    print ("Interconnections removed.\n")
    return skl_int, int_3

def data_labels(skl_int):
    """It calculate the position of the centroid for every 2D segment and returns: centroid position (z, y, x), label value and its orientation"""
    print ("\nAcquiring data...")
    s = np.ones((3, 3))
    lbl = np.zeros((skl_int.shape))
    data_lbl = []
    n0=0
    for i in range(len(skl_int)):
        lb, n = scipy.ndimage.measurements.label(skl_int[i], structure=s)
        lbl[i]=lb
        loc = scipy.ndimage.find_objects(lb, max_label=n)
        for k in range(0,n,1):
            label_object = lb[loc[k]].copy()
            label_object[label_object!=k+1]=0
            y, x = np.nonzero(label_object)
            y0 = loc[k][0].start
            x0 = loc[k][1].start
            length = float(len(x))
            c_y, c_x = round(np.sum(y)/length, 0), round(np.sum(x)/length, 0)
            ang = svd_or(x, y)
            data_lbl.append([i, int(c_y+y0), int(c_x+x0), k+1+n0, int(ang)])
        n0+=n
    data_lbl = np.array(data_lbl, copy=False)
    lbl0 = lbl.copy()
    for i in range (len(lbl0)):
        if i>0:
            lbl0[i][lbl0[i]>0]+=np.amax(lbl0[i-1])
    lbl = lbl0
    print ("Format data created:\ncentroid_z, centroid_y, centroid_x, label_value, label_orient")
    return data_lbl, lbl

def svd_or(x, y):
    """Calculate the orientation of the 2D label using SVD"""
    north = [0,1]
    x = x - np.mean(x)
    y = y - np.mean(y); y = y[::-1]
    _, _, vv = np.linalg.svd(np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1))
    v = vv[0]
    v = v*(-1.0 if v[1]>=0.0 else 1.0)
    v1_u = north/np.linalg.norm(north)
    v2_u = v/np.linalg.norm(v)
    ang = round(np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))),0)
    ang = 360-ang if v[0]<0 else ang
    return ang

def remove_spots(img, size, int_3):
    """Remove labels with less than 'size' elements (artifacts from skeletonization)"""
    print ("\nRemoving small elements of size <= {} px...".format(size))
    lbl = np.zeros((img.shape))
    s = np.ones((3, 3))
    for i in range(0,len(lbl),1):
        lb, _ = scipy.ndimage.measurements.label(img[i], structure=s)
        lbl[i]=lb
    indx_to_remove = []
    for n, i in enumerate(lbl):
        lbl_val = np.unique(i[i.nonzero()])
        for k in lbl_val:
            y, x = np.where(i==k)
            if len(x)<=size:
                z = np.full((len(x)), n)
                lbl[z, y, x]=0
                int_3_n = np.where(int_3[:,0]==n)[0]
                indx, d = indx_nearest_neighbours(np.column_stack((y, x)), int_3[int_3_n,1:], 1)
    lbl[lbl>0]=1
    int_3_c = np.delete(int_3, indx_to_remove, axis=0)
    print ("Small elements removed. {} in total.\n".format(len(int_3)-len(int_3_c)))
    return lbl, int_3_c

def couple_at_intersections(lbl, vx, data_lbl, int_3, tol_or, small_segm, tol_dist_i, loc_tol_or):
    """It pairs segments at intersections using different criteria"""
    print ("\nPairing labels at intersections...")
    n_img = 0
    eg=vx//2
    lbl_c = lbl.copy()
    data_lbl_c = data_lbl.copy()
    el_anlz = []
    i_prev_all = []
    for i in int_3:
        if n_img!=i[0]:
            print ("Image {} of {}.".format(i[0], len(lbl)-1))
        n_img = i[0]
        vox = lbl_c[i[0], i[1]-eg:i[1]+vx-eg, i[2]-eg:i[2]+vx-eg].copy()
        fracts = np.unique(vox[vox.nonzero()])
        indx = np.array([np.where((data_lbl_c[:,0]==i[0]) & (data_lbl_c[:,3]==k))[0][0] for k in fracts])
        if len (fracts)>1:
            or_label = data_lbl_c[indx,-1]
            vals_vox = np.unique(vox[vox>0])
            couples = np.array(list(combinations(vals_vox, 2)))
            check = []
            if len(np.in1d(couples, np.array(el_anlz), invert=True).nonzero()[0])==0:
                indx=[]
            else:
                for el in couples:
                    if len(el_anlz)>1:
                        if any(np.equal(el_anlz, el[el.argsort()]).all(1))==False:
                            el_anlz.append(el[el.argsort()])
                        else:
                            check.append(el[el.argsort()])
                    else:
                        el_anlz.append(el[el.argsort()])
            if len(indx)>1:
                size = [len(np.where(lbl_c[i[0]]==el)[0]) for el in fracts]
                if min(size)>=small_segm[0]:
                    flat = (265,266,267,268,269,270,90,91,92,93,94,95)
                else:
                    flat = (263,264,265,266,267,268,269,270,90,91,92,93,94,95,96,97)
                if any(el in or_label for el in flat):
                    if len(or_label[(or_label>=90) & (or_label<=flat[-1])])==1:
                        if len(np.where((or_label>=90) & (or_label<=90+tol_or))[0])==1:
                            indx_t = np.where(or_label<=flat[-1])[0]
                            t = data_lbl_c[indx[indx_t]][0].copy()
                            t[-1]=270
                            data_lbl_c[indx[indx_t]]=t
                            or_label[or_label<=flat[-1]]=270
                    elif len(or_label[(or_label>=flat[0]) & (or_label<=270)])==1:
                        if len(np.where((or_label<=270) & (or_label>=270-tol_or))[0])==1:
                            indx_t = np.where(or_label>=flat[0])[0]
                            t = data_lbl_c[indx[indx_t]][0].copy()
                            t[-1]=90
                            data_lbl_c[indx[indx_t]]=t
                            or_label[or_label>=flat[0]]=90
                    else:
                        pass
                result_indexes = []
                for idx1,idx2 in combinations(range(len(or_label)),2):
                    or_diff = abs(or_label[idx1]-or_label[idx2])
                    size_a = len(np.where(lbl_c==data_lbl_c[indx[idx1], -2])[0])
                    size_b = len(np.where(lbl_c==data_lbl_c[indx[idx2], -2])[0])
                    small = size_a<=small_segm[0] or size_b<=small_segm[0]
                    if small:
                        if or_diff <= tol_or+(tol_or/1.5):
                            result_indexes.append((idx1,idx2, or_diff))
                        elif min(size_a, size_b)<=small_segm[0] and any(or_lbl in (270, 90) for or_lbl in (or_label[idx1], or_label[idx2])):
                            result_indexes.append((idx1,idx2, or_diff))
                        elif min(size_a, size_b)==2:
                            result_indexes.append((idx1,idx2, or_diff))
                        else:
                            pass
                    else:
                        if or_diff <= tol_or:
                            result_indexes.append((idx1,idx2, or_diff))
                if result_indexes:
                    result_indexes = np.array(result_indexes)
                    result_indexes = result_indexes[result_indexes[:,-1].argsort()]
                    result_indexes= save_continuity_if_parallel (result_indexes, or_label, i, lbl_c, data_lbl_c, data_lbl, indx, int_3, vx, eg)
                    if check:
                        if len(np.in1d(data_lbl_c[indx[np.unique(result_indexes[:,:2])], -2], np.unique(check)).nonzero()[0])>0:
                            result_indexes = check_labels (i, check, indx, data_lbl, data_lbl_c, result_indexes, lbl_c, lbl)
                    if len(fracts)==3 and len(result_indexes)==2 and len(np.unique(result_indexes[:,:2]))==3:
                        result_indexes = bifurcation_correction (i, result_indexes, indx, data_lbl_c, lbl_c)
                        if any(or_el in data_lbl_c[indx[np.unique(result_indexes[:,:2])],-1] for or_el in (90, 270)):
                            pass
                        else:
                            if len(np.where(result_indexes[:,-1]>loc_tol_or)[0])<=1:
                                result_indexes = result_indexes[result_indexes[:,-1]<=loc_tol_or]
                        if i[0]>0 and len(int_3[np.where(int_3[:,0]==i[0]-1)[0],1:])>0 and len(result_indexes)==2:
                            result_indexes, i_prev = save_3d_continuity(fracts, result_indexes, i, int_3, tol_or, data_lbl_c, data_lbl, lbl_c, lbl, indx, vx, eg, tol_dist_i, i_prev_all)
                            if i_prev:
                                i_prev_all.append (i_prev[0])
                    u=[]
                    for j in result_indexes:
                        if j[0] not in u and j[1] not in u:
                            u.append(j[0]); u.append(j[1])
                    u = np.reshape(u, (len(u)//2, 2))
                    for w in u:
                        mx = max (data_lbl[indx[w[0]],3], data_lbl[indx[w[1]],3])
                        mn = min (data_lbl[indx[w[0]],3], data_lbl[indx[w[1]],3])
                        lbl[lbl==mx]=mn
                        for g in np.where(data_lbl[:,3]==mx)[0]:
                            t = data_lbl[g].copy(); t[3]=mn
                            data_lbl[g]=t
    print ("Labels coupled.\n")
    return lbl, data_lbl, data_lbl_c

def bifurcation_correction (i, result_indexes, indx, data_lbl_c, lbl_c):
    """Estimate orientation in proximity of the interconnection using the closest 15 pixels of the segment"""
    result_indexes_2 = []
    for el in result_indexes:
        val_el0 = data_lbl_c[indx[el[0]]]
        val_el1 = data_lbl_c[indx[el[1]]]
        av_or = (float(val_el0[-1])+float(val_el1[-1]))/2
        if 135<av_or<225:
            indx_a = np.where(lbl_c[i[0]]==val_el0[-2])
            indx_a = np.column_stack((indx_a[0], indx_a[1]))
            size_a = 15 if len(indx_a)>5 else len(indx_a)
            indx_a = indx_a[indx_a[:,1].argsort()]
            in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
            cl_a = indx_a[in_a, 1]
            v_a = indx_a[:size_a] if abs(cl_a-indx_a[0,1])<abs(cl_a-indx_a[-1,1]) else indx_a[-size_a:]
            or_a = svd_or(v_a[:,1], v_a[:,0])
            indx_b = np.where(lbl_c[i[0]]==val_el1[-2])
            indx_b = np.column_stack((indx_b[0], indx_b[1]))
            size_b = 15 if len(indx_b)>5 else len(indx_b)
            indx_b = indx_b[indx_b[:,0].argsort()]
            in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
            cl_b = indx_b[in_b, 1]
            v_b = indx_b[:size_b] if abs(cl_b-indx_b[0,1])<abs(cl_b-indx_b[-1,1]) else indx_b[-size_b:]
            or_b = svd_or(v_b[:,1], v_b[:,0])
            result_indexes_2.append([el[0], el[1], int(abs(or_a-or_b))])
        else:
            indx_a = np.where(lbl_c[i[0]]==val_el0[-2])
            indx_a = np.column_stack((indx_a[0], indx_a[1]))
            size_a = 15 if len(indx_a)>5 else len(indx_a)
            indx_a = indx_a[indx_a[:,0].argsort()]
            in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
            cl_a = indx_a[in_a, 0]
            v_a = indx_a[:size_a] if abs(cl_a-indx_a[0,0])<abs(cl_a-indx_a[-1,0]) else indx_a[-size_a:]
            or_a = svd_or(v_a[:,1], v_a[:,0])
            indx_b = np.where(lbl_c[i[0]]==val_el1[-2])
            indx_b = np.column_stack((indx_b[0], indx_b[1]))
            size_b = 15 if len(indx_b)>5 else len(indx_b)
            indx_b = indx_b[indx_b[:,1].argsort()]
            in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
            cl_b = indx_b[in_b, 0]
            v_b = indx_b[:size_b] if abs(cl_b-indx_b[0,0])<abs(cl_b-indx_b[-1,0]) else indx_b[-size_b:]
            or_b = svd_or(v_b[:,1], v_b[:,0])
            result_indexes_2.append([el[0], el[1], int(abs(or_a-or_b))])
    result_indexes_2 = np.array(result_indexes_2)
    result_indexes_2 = result_indexes_2[np.argsort(np.ravel(result_indexes_2[:,-1]))]
    return result_indexes_2

def save_3d_continuity(fracts, result_indexes, i, int_3, tol_or, data_lbl_c, data_lbl, lbl_c, lbl, indx, vx, eg, tol_dist_i, i_prev_all):
    """In bifurcations, preserve the connection in 3D"""
    vx2 = vx*2
    eg2 = eg*2
    vox_b = lbl_c[i[0], i[1]-eg2:i[1]+vx2-eg2, i[2]-eg2:i[2]+vx2-eg2].copy()
    cs = []
    for el0 in fracts:
        y, x = np.where(vox_b==el0)
        length = float(len(x))
        c_y, c_x = int(round(np.sum(y)/length, 0)), int(round(np.sum(x)/length, 0))
        cs.append([el0, c_y, c_x])
    cs = np.array(cs, dtype=int, copy=False)
    i_prev_indx = np.where(int_3[:,0]==i[0]-1)[0]
    indx_i_prev_img, d_i = indx_nearest_neighbours(i[1:], int_3[i_prev_indx,1:], 1)
    d_i_check = []
    if d_i<=tol_dist_i:
        i_current_image = int_3[int_3[:,0]==i[0], 1:]
        indx_check, d_check = indx_nearest_neighbours(int_3[i_prev_indx[indx_i_prev_img],1:], i_current_image, 1)
        if i[1]==i_current_image[indx_check, 0] and i[2]==i_current_image[indx_check, 1]:
            pass
        else:
            if round(d_i, 0)<=round(d_check, 0):
                pass
            else:
                if abs(round(d_i, 0)-round(d_check, 0))<=2:
                    pass
                else:
                    d_i_check.append(tol_dist_i+1)
    i_prev_indx = i_prev_indx[indx_i_prev_img]
    i_prev_record = []
    d_i = round(d_i, 0)
    distance_thr = d_i<=tol_dist_i
    i_prev_n = i_prev_indx not in i_prev_all
    if d_i_check:
        distance_thr_check=False
    else:
        distance_thr_check = distance_thr
    if distance_thr_check and i_prev_n:
        i_prev = int_3[i_prev_indx]
        vox_prev = lbl_c[i_prev[0], i_prev[1]-eg:i_prev[1]+vx-eg, i_prev[2]-eg:i_prev[2]+vx-eg].copy()
        fracts_prev = np.unique(vox_prev[vox_prev.nonzero()])
        if len(fracts_prev)==3:
            vox_b_prev =lbl_c[i_prev[0], i_prev[1]-eg2:i_prev[1]+vx2-eg2, i_prev[2]-eg2:i_prev[2]+vx2-eg2].copy()
            cs_prev = []
            for el_prev in fracts_prev:
                y, x = np.where(vox_b_prev==el_prev)
                length = float(len(x))
                c_y, c_x = int(round(np.sum(y)/length, 0)), int(round(np.sum(x)/length, 0))
                cs_prev.append([el_prev, c_y, c_x])
            cs_prev = np.array(cs_prev, dtype=int, copy=False)
            indx_coup = np.ravel(result_indexes[:,:2])
            indx_val0 = np.unique([el for el in indx_coup if len(indx_coup[indx_coup==el])==1])
            val0_0 = data_lbl_c[indx[indx_val0[np.in1d(indx_val0, result_indexes[0,:2])]],-2][0]
            val_0_indx, d0 = indx_nearest_neighbours(cs[cs[:,0]==val0_0,1:][0], cs_prev[:,1:], 1)
            indx_val1 = np.unique([el for el in indx_coup if len(indx_coup[indx_coup==el])==2])
            val1 = data_lbl_c[indx[indx_val1], -2][0]
            val_1_indx, d1 = indx_nearest_neighbours(cs[cs[:,0]==val1,1:][0], cs_prev[:,1:], 1)
            val0_1 = data_lbl_c[indx[np.in1d(np.ravel(data_lbl_c[indx,-2]), np.array([val0_0, val1]), invert=True)],-2][0]
            val_0_1_indx, d0_1 = indx_nearest_neighbours(cs[cs[:,0]==val0_1,1:][0], cs_prev[:,1:], 1)
            if cs_prev[val_0_indx,0]==cs_prev[val_0_1_indx,0]:
                if d0<=d0_1:
                    pass
                else:
                    val_indx_all = np.arange(0,len(cs_prev),1)
                    val_0_indx = val_indx_all[np.in1d(val_indx_all, np.array([val_0_indx, val_1_indx]), invert=True)][0]
            if max(d1, d0)<=tol_dist_i//2 and val_0_indx!=val_1_indx:
                i_prev_record.append(i_prev_indx)
                if data_lbl[data_lbl_c[:,-2]==cs_prev[val_1_indx, 0],-2]!=data_lbl[data_lbl_c[:,-2]==cs_prev[val_0_indx, 0],-2]:
                    result_indexes = np.delete(result_indexes, 0, axis=0)
                else:
                    result_indexes = np.delete(result_indexes, 1, axis=0)
    not_analyzed = len(result_indexes)>1
    small_or = any(or_el in data_lbl_c[indx[np.unique(result_indexes[:,:2])],-1] for or_el in (90, 270))
    if not_analyzed:
        small_seg = any(el2 for el2 in [len(lbl_c[lbl_c==el]) for el in data_lbl_c[indx[np.unique(result_indexes[:,:2])],-2]] if el2<6)
    else:
        small_seg = False
    if not_analyzed and not distance_thr and not small_or and not small_seg:
        vox_b_prev = lbl[i[0]-1, i[1]-eg2:i[1]+vx2-eg2, i[2]-eg2:i[2]+vx2-eg2].copy()
        if len(np.unique(vox_b_prev[vox_b_prev>0]))>0:
            lbls_prev=[]
            coords_prev = np.column_stack((vox_b_prev.nonzero()))
            for n2, el2 in enumerate(result_indexes):
                el2_0 = data_lbl_c[indx[el2[0]]]
                el2_0_c = cs[cs[:,0]==el2_0[-2]][0]
                el2_1 = data_lbl_c[indx[el2[1]]]
                el2_1_c = cs[cs[:,0]==el2_1[-2]][0]
                indx_0, d0 = indx_nearest_neighbours(el2_0_c[1:], coords_prev, 1)
                indx_1, d1 = indx_nearest_neighbours(el2_1_c[1:], coords_prev, 1)
                y0, x0 = coords_prev[indx_0]
                y1, x1 = coords_prev[indx_1]
                if vox_b_prev[y0, x0]==vox_b_prev[y1, x1]:
                    lbls_prev.append([n2, (d0+d1)/2])
            if len(lbls_prev)>0:
                if len(lbls_prev)==1:
                    result_indexes = np.array([result_indexes[lbls_prev[0][0]]], dtype=int, copy=False)
                else:
                    lbls_prev = np.array(lbls_prev, copy=False)
                    lbls_prev = lbls_prev[lbls_prev[:,1].argsort()]
                    result_indexes = np.delete(result_indexes,int(lbls_prev[1,0]), axis=0)
            else:
                pass
        else:
            pass
    return result_indexes, i_prev_record

def check_labels (i, check, indx, data_lbl, data_lbl_c, result_indexes, lbl_c, lbl):
    """If a label has been already paired, it checks for parallelisms (the segments respect the criteria but lay on different planes)"""
    mtch = []
    for pos, el in enumerate(result_indexes):
        indx_el0 = indx[el[0]]
        indx_el1 = indx[el[1]]
        check_el0_c = len(np.where(data_lbl_c[:,-2]==data_lbl_c[indx_el0, -2])[0])
        check_el0 = len(np.where(data_lbl[:,-2]==data_lbl[indx_el0, -2])[0])
        check_el1_c = len(np.where(data_lbl_c[:,-2]==data_lbl_c[indx_el1, -2])[0])
        check_el1 = len(np.where(data_lbl[:,-2]==data_lbl[indx_el1, -2])[0])
        if data_lbl[indx_el0, -2] != data_lbl[indx_el1, -2]:
            if check_el0_c != check_el0:
                val0 = data_lbl[indx_el0, -2]
                y_val0, x_val0 = np.where(lbl[i[0]]==val0)
                val0_ang = svd_or(x_val0, y_val0)
                indx_val0 = np.where(data_lbl[:,-2]==val0)[0]
                lbl1 = data_lbl_c[indx_el1]
                av_or = (float(val0_ang)+float(lbl1[-1]))/2
                for lbl0 in data_lbl_c[indx_val0]:
                    n = check_if_parallel_2 (lbl1, lbl0, i, lbl_c, av_or)
                    if n>0:
                        mtch.append(pos)
                        break
            elif check_el1_c != check_el1:
                val1 = data_lbl[indx_el1, -2]
                y_val1, x_val1 = np.where(lbl[i[0]]==val1)
                val1_ang = svd_or(x_val1, y_val1)
                indx_val1 = np.where(data_lbl[:,-2]==val1)[0]
                lbl0 = data_lbl_c[indx_el0]
                av_or = (float(val1_ang)+float(lbl0[-1]))/2
                for lbl1 in data_lbl_c[indx_val1]:
                    n = check_if_parallel_2 (lbl0, lbl1, i, lbl_c, av_or)
                    if n>0:
                        mtch.append(pos)
                        break
            else:
                pass
        else:
            mtch.append(pos)
    mtch = np.array(mtch)
    if len(mtch)>0:
        result_indexes = result_indexes[np.in1d(np.arange(0, len(result_indexes),1), mtch, invert=True)]
    return result_indexes

def check_if_parallel_2 (lbl1, lbl2, i, lbl_c, av_or):
    """It returns a positive value for parallel labels"""
    n = []
    if 135<av_or<225:
        indx_a = np.where(lbl_c[i[0]]==lbl1[-2])
        indx_a = np.column_stack((indx_a[0], indx_a[1]))
        indx_a = indx_a[indx_a[:,0].argsort()]
        in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
        cl_a = indx_a[in_a, 0]
        v_a = indx_a[0,0]-indx_a[-1,0] if abs(cl_a-indx_a[0,0])<abs(cl_a-indx_a[-1,0]) else indx_a[-1,0]-indx_a[0,0]
        indx_b = np.where(lbl_c[i[0]]==lbl2[-2])
        indx_b = np.column_stack((indx_b[0], indx_b[1]))
        indx_b = indx_b[indx_b[:,0].argsort()]
        in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
        cl_b = indx_b[in_b, 0]
        v_b = indx_b[0,0]-indx_b[-1,0] if abs(cl_b-indx_b[0,0])<abs(cl_b-indx_b[-1,0]) else indx_b[-1,0]-indx_b[0,0]
        n.append(v_a*v_b)
    else:
        indx_a = np.where(lbl_c[i[0]]==lbl1[-2])
        indx_a = np.column_stack((indx_a[0], indx_a[1]))
        indx_a = indx_a[indx_a[:,1].argsort()]
        in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
        cl_a = indx_a[in_a, 1]
        v_a = indx_a[0,1]-indx_a[-1,1] if abs(cl_a-indx_a[0,1])<abs(cl_a-indx_a[-1,1]) else indx_a[-1,1]-indx_a[0,1]
        indx_b = np.where(lbl_c[i[0]]==lbl2[-2])
        indx_b = np.column_stack((indx_b[0], indx_b[1]))
        indx_b = indx_b[indx_b[:,1].argsort()]
        in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
        cl_b = indx_b[in_b, 1]
        v_b = indx_b[0,1]-indx_b[-1,1] if abs(cl_b-indx_b[0,1])<abs(cl_b-indx_b[-1,1]) else indx_b[-1,1]-indx_b[0,1]
        n.append(v_a*v_b)
    return n[0]

def check_if_parallel(j, or_label, i, lbl_c, data_lbl_c, indx):
    """It returns a positive value for parallel labels"""
    av_or = (float(or_label[j[0]])+float(or_label[j[1]]))/2
    n = []
    if 135<av_or<225:
        indx_a = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[0]], -2])
        indx_a = np.column_stack((indx_a[0], indx_a[1]))
        indx_a = indx_a[indx_a[:,0].argsort()]
        in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
        cl_a = indx_a[in_a, 0]
        v_a = indx_a[0,0]-indx_a[-1,0] if abs(cl_a-indx_a[0,0])<abs(cl_a-indx_a[-1,0]) else indx_a[-1,0]-indx_a[0,0]
        indx_b = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[1]], -2])
        indx_b = np.column_stack((indx_b[0], indx_b[1]))
        indx_b = indx_b[indx_b[:,0].argsort()]
        in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
        cl_b = indx_b[in_b, 0]
        v_b = indx_b[0,0]-indx_b[-1,0] if abs(cl_b-indx_b[0,0])<abs(cl_b-indx_b[-1,0]) else indx_b[-1,0]-indx_b[0,0]
        n.append(v_a*v_b)
    else:
        indx_a = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[0]], -2])
        indx_a = np.column_stack((indx_a[0], indx_a[1]))
        indx_a = indx_a[indx_a[:,1].argsort()]
        in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
        cl_a = indx_a[in_a, 1]
        v_a = indx_a[0,1]-indx_a[-1,1] if abs(cl_a-indx_a[0,1])<abs(cl_a-indx_a[-1,1]) else indx_a[-1,1]-indx_a[0,1]
        indx_b = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[1]], -2])
        indx_b = np.column_stack((indx_b[0], indx_b[1]))
        indx_b = indx_b[indx_b[:,1].argsort()]
        in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
        cl_b = indx_b[in_b, 1]
        v_b = indx_b[0,1]-indx_b[-1,1] if abs(cl_b-indx_b[0,1])<abs(cl_b-indx_b[-1,1]) else indx_b[-1,1]-indx_b[0,1]
        n.append(v_a*v_b)
    return n[0]

def check_if_parallel_inv(j, or_label, i, lbl_c, data_lbl_c, indx, excp):
    """It returns a positive value for parallel labels"""
    n=[]
    if len(excp)==0:
        y_c = i[1]; x_c = i[2]
        lbl1 = data_lbl_c[indx[j[0]]]
        lbl2 = data_lbl_c[indx[j[1]]]
        v_a = (y_c-lbl1[1])*(x_c-lbl1[2])
        v_b = (y_c-lbl2[1])*(x_c-lbl2[2])
        n.append(v_a*v_b*-1)
    else:
        av_or = (float(or_label[j[0]])+float(or_label[j[1]]))/2
        if 135<av_or<225:
            indx_a = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[0]], -2])
            indx_a = np.column_stack((indx_a[0], indx_a[1]))
            size_a = len(indx_a)//5 if len(indx_a)//5>=10 else -1
            indx_a = indx_a[indx_a[:,0].argsort()]
            in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
            cl_a = indx_a[in_a, 1]
            v_a = indx_a[0,1]-indx_a[size_a,1] if abs(cl_a-indx_a[0,1])<abs(cl_a-indx_a[-1,1]) else indx_a[-1,1]-indx_a[-size_a,1]
            indx_b = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[1]], -2])
            indx_b = np.column_stack((indx_b[0], indx_b[1]))
            size_b = len(indx_b)//5 if len(indx_b)//5>=10 else -1
            indx_b = indx_b[indx_b[:,0].argsort()]
            in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
            cl_b = indx_b[in_b, 1]
            v_b = indx_b[0,1]-indx_b[size_b,1] if abs(cl_b-indx_b[0,1])<abs(cl_b-indx_b[-1,1]) else indx_b[-1,1]-indx_b[-size_b,1]
            n.append(v_a*v_b)
        else:
            indx_a = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[0]], -2])
            indx_a = np.column_stack((indx_a[0], indx_a[1]))
            size_a = len(indx_a)//5 if len(indx_a)//5>=10 else -1
            indx_a = indx_a[indx_a[:,1].argsort()]
            in_a = indx_nearest_neighbours(i[1:], indx_a, 1)[0]
            cl_a = indx_a[in_a, 0]
            v_a = indx_a[0,0]-indx_a[size_a,0] if abs(cl_a-indx_a[0,0])<abs(cl_a-indx_a[-1,0]) else indx_a[-1,0]-indx_a[-size_a,0]
            indx_b = np.where(lbl_c[i[0]]==data_lbl_c[indx[j[1]], -2])
            indx_b = np.column_stack((indx_b[0], indx_b[1]))
            size_b = len(indx_b)//5 if len(indx_b)//5>=10 else -1
            indx_b = indx_b[indx_b[:,1].argsort()]
            in_b = indx_nearest_neighbours(i[1:], indx_b, 1)[0]
            cl_b = indx_b[in_b, 0]
            v_b = indx_b[0,0]-indx_b[size_b,0] if abs(cl_b-indx_b[0,0])<abs(cl_b-indx_b[-1,0]) else indx_b[-1,0]-indx_b[-size_b,0]
            n.append(v_a*v_b)
    return n[0]

def check_exeption(j, i, data_lbl_c, indx):
    """It checks for parallel labels in the same quadrant"""
    y_c = i[1]; x_c = i[2]
    lbl1 = data_lbl_c[indx[j[0]]]
    lbl2 = data_lbl_c[indx[j[1]]]
    y1 = y_c-lbl1[1]; x1 = x_c-lbl1[2]
    y2 = y_c-lbl2[1]; x2 = x_c-lbl2[2]
    excp = []
    if y1*y2>=0 and x1*x2>=0:
        excp.append(1)
    return excp

def save_continuity_if_parallel (result_indexes, or_label, i, lbl_c, data_lbl_c, data_lbl, indx, int_3, vx, eg):
    """It preserves the continuity when parallel labels occurr in the middle of a fracture"""
    if len(indx)==3:
        ind=[]
        ind_np = []
        for ind_, k in enumerate(result_indexes):
            n = check_if_parallel(k, or_label, i, lbl_c, data_lbl_c, indx)
            if n>0:
                ind.append(ind_)
            if n<0:
                ind_np.append(ind_)
        ind_np = np.array(ind_np, dtype=int, copy=False)
        if len(ind_np)==2 and len(ind)==1:
            j = result_indexes[ind][0].copy()
            j0_c = data_lbl_c[indx[j[0]], -2]; j0 = data_lbl[indx[j[0]], -2]
            j1_c = data_lbl_c[indx[j[1]], -2]; j1 = data_lbl[indx[j[1]], -2]
            if j0_c == j0 and j1_c != j1:
                for i2 in int_3[int_3[:,0]==i[0]]:
                    vox_check = lbl_c[i2[0], i2[1]-eg:i2[1]+vx-eg, i2[2]-eg:i2[2]+vx-eg]
                    vals = np.unique(vox_check[vox_check>0])
                    j0_c_in = j0_c in vals
                    j1_c_in = j1_c in vals
                    i_different = not np.array_equal(i, i2)
                    if j0_c_in and j1_c_in and i_different:
                        ind_np = ind_np[np.in1d(ind_np, np.where(result_indexes[:,:2]==j[1])[0])]
                        break
                result_indexes = result_indexes[ind_np]
            elif j0_c != j0 and j1_c == j1:
                for i2 in int_3[int_3[:,0]==i[0]]:
                    vox_check = lbl_c[i2[0], i2[1]-eg:i2[1]+vx-eg, i2[2]-eg:i2[2]+vx-eg]
                    vals = np.unique(vox_check[vox_check>0])
                    j0_c_in = j0_c in vals
                    j1_c_in = j1_c in vals
                    i_different = not np.array_equal(i, i2)
                    if j0_c_in and j1_c_in and i_different:
                        ind_np = ind_np[np.in1d(ind_np, np.where(result_indexes[:,:2]==j[0])[0])]
                        break
                result_indexes = result_indexes[ind_np]
            else:
                result_indexes = result_indexes[ind_np]
        else:
            result_indexes = result_indexes[ind_np]
    elif len(indx)==4:
        ind_np = []
        excp = []
        for ind_, k in enumerate(result_indexes):
            n = check_if_parallel(k, or_label, i, lbl_c, data_lbl_c, indx)
            if n<=0:
                ind_np.append(ind_)
            else:
                excpt = check_exeption(k, i, data_lbl_c, indx)
                if excpt:
                    excp.append(excpt)
        ind_np = np.array(ind_np, dtype=int, copy=False)
        result_indexes = result_indexes[ind_np]
        ind_np2=[]
        for ind_2, k in enumerate(result_indexes):
            n = check_if_parallel_inv(k, or_label, i, lbl_c, data_lbl_c, indx, excp)
            if n<=0:
                ind_np2.append(ind_2)
        if len(ind_np2)>0:
            ind_np2 = np.array(ind_np2, dtype=int, copy=False)
            result_indexes = result_indexes[ind_np2]
    else:
        ind_np=[]
        for ind_, k in enumerate(result_indexes):
            n = check_if_parallel(k, or_label, i, lbl_c, data_lbl_c, indx)
            if n<0:
                ind_np.append(ind_)
        result_indexes = result_indexes[ind_np]
    return result_indexes

def rearrange_values (lbl):
    """It rearranges labels' value from 1 to len(num_of_labels) to show better the analysis"""
    b = np.unique(lbl[lbl>0])
    a = np.array(range(1,len(b)+1))
    indexes = []
    for i in range(0,len(a),1):
        indexes.append(np.where(lbl==b[i]))
    for i, k in enumerate (indexes):
        lbl[k]=a[i]
    return lbl

def data_labels_coupled (lbl):
    """It measures the new centroid and orientation for the paired segments"""
    print ("\nAcquiring data for the coupled labels...")
    lbl = rearrange_values (lbl)
    data_lbl = []
    for n, i in enumerate(lbl):
        labels = np.unique(i[i>0])
        for k in labels:
            y, x = np.where(i==k)
            length = float(len(x))
            c_y, c_x = round(np.sum(y)/length, 0), round(np.sum(x)/length, 0)
            ang = svd_or(x, y)
            data_lbl.append([n, int(c_y), int(c_x), k, int(ang)])
    data_lbl = np.array(data_lbl, copy=False)
    print ("Done\n")
    return data_lbl, lbl

def correct_flat_ang (i, ind, tol_or):
    """Check for sub-horizontal element bias and correct it"""
    flat = [264,265,266,267,268,269,270,90,91,92,93,94,95,96]
    if ind[-1] in (flat):
        if ind[-1]>=flat[0]:
            if 90<=i[-1]<=90+tol_or:
                ind[-1]=90
        elif ind[-1]<=flat[-1]:
            if 270-tol_or<=i[-1]<=270:
                ind[-1]=270
        else:
            pass
    elif i[-1] in (flat):
        if i[-1]>=flat[0]:
            if 90<=ind[-1]<=90+tol_or:
                i[-1]=90
        elif i[-1]<=flat[-1]:
            if 270-tol_or<=ind[-1]<=270:
                i[-1]=270
        else:
            pass
    else:
        pass
    return i, ind

def connect_labels_3d (data_lbl, data_lbl_c, lbl_coupled, lbl_c, tol_sp_base, tol_or_base, small_segm, int_3, vx):
    """It connect labels in 3D and tries to connect non-connected labels using the old data"""
    eg = vx//2
    vx2 = vx*2
    eg2 = eg*2
    all_elements = len(np.unique(lbl_coupled))
    not_connected=0
    data_lbl_2 = data_lbl.copy()
    for k in range(len(data_lbl[data_lbl[:,0]<np.amax(data_lbl[:,0])])):
        i = data_lbl[data_lbl[:,0]<np.amax(data_lbl[:,0])][k]
        c = data_lbl[data_lbl[:,0]==i[0]+1]
        coords_lbl = np.column_stack(np.where(lbl_coupled[int(i[0])]==i[-2]))
        if len(coords_lbl)>=5:
            indx_lbl_pxs, d_pxs = indx_nearest_neighbours(i[1:3], coords_lbl, 5)
        else:
            indx_lbl_pxs, d_pxs = indx_nearest_neighbours(i[1:3], coords_lbl, len(coords_lbl))
        pxs_lbl_i = coords_lbl[indx_lbl_pxs]
        cy_lbl_i= int(((np.amax(pxs_lbl_i[:,0])-np.amin(pxs_lbl_i[:,0]))/2)+np.amin(pxs_lbl_i[:,0]))
        cx_lbl_i = int(((np.amax(pxs_lbl_i[:,1])-np.amin(pxs_lbl_i[:,1]))/2)+np.amin(pxs_lbl_i[:,1]))
        vox_next = lbl_coupled[int(i[0])+1, cy_lbl_i-eg2:cy_lbl_i+vx2-eg2, cx_lbl_i-eg2:cx_lbl_i+vx2-eg2].copy()
        vals_vox_next = np.unique(vox_next[vox_next>0])
        if len(vals_vox_next)==0:
            ind = []
        elif len(vals_vox_next)==1:
            ind = data_lbl[np.where((data_lbl[:,0]==i[0]+1) & (data_lbl[:,-2]==vals_vox_next[0]))][0]
        else:
            coords_nxt_y, coords_nxt_x = np.where(vox_next>0)
            coords_nxt_y = coords_nxt_y+(cy_lbl_i-eg2)
            coords_nxt_x = coords_nxt_x+(cx_lbl_i-eg2)
            coords_nxt = np.column_stack((coords_nxt_y, coords_nxt_x))
            lbl_nxt = []
            for el in pxs_lbl_i:
                el_nxt, _ = indx_nearest_neighbours(el, coords_nxt, 1)
                lbl_nxt.append(lbl_coupled[int(i[0]+1), coords_nxt[el_nxt,0], coords_nxt[el_nxt,1]])
            if len(np.unique(lbl_nxt))==1:
                ind = data_lbl[np.where((data_lbl[:,0]==i[0]+1) & (data_lbl[:,-2]==lbl_nxt[0]))][0]
            else:
                lbl_nxt2 = []
                for el2 in np.unique(lbl_nxt):
                    lbl_nxt2.append([el2, len(np.where(lbl_nxt==el2)[0])])
                lbl_nxt2 = np.array(lbl_nxt2, copy=False)
                lbl_nxt2 = lbl_nxt2[lbl_nxt2[:,1].argsort()]
                ind = data_lbl[np.where((data_lbl[:,0]==i[0]+1) & (data_lbl[:,-2]==lbl_nxt2[-1,0]))][0]
        if len(ind)>0:
            indx = np.where(c[:,-2]==ind[-2])[0][0]
            dist = np.sqrt((i[1]-ind[1])**2+(i[2]-ind[2])**2)
        else:
            indx, dist = indx_nearest_neighbours(i[1:3], c[:,1:3], 1)
            ind = data_lbl[data_lbl[:,0]==i[0]+1][indx]
        size_a = len(np.where(lbl_coupled[int(i[0])]==i[-2])[0])
        size_b = len(np.where(lbl_coupled[int(c[indx][0])]==c[indx][-2])[0])
        sizes = float(size_a+size_b)/2.0
        tol_sp = int(sizes/3.7) if sizes>100 else int(sizes/2.5)
        ind_check = data_lbl_2[np.where((data_lbl_2[:,0]==ind[0]) & (data_lbl_2[:,1]==ind[1]) & (data_lbl_2[:,2]==ind[2]))][0]
        if ind_check[-2]!=ind[-2]:
            n, dist = check_if_continuous (i, ind, lbl_coupled)
            size_b = cp.copy(size_a)
        else:
            n=-1
        new_ind=[]
        if (max(size_a, size_b)>=small_segm[0]*15 and float(min(size_a, size_b))<=float(max(size_a, size_b))/3) or dist>tol_sp or n>0:
            indx_2, dist_2 = indx_nearest_neighbours(i[1:3], c[:,1:3], 5)
            indx_2 = indx_2[np.where(dist_2<=tol_sp_base)]
            check_ind = []
            if len(indx_2)>0:
                for el in indx_2:
                    size_b2 = len(np.where(lbl_coupled[int(c[el][0])]==c[el][-2])[0])
                    el_lbl = data_lbl[data_lbl[:,0]==i[0]+1][el]
                    if abs(el_lbl[-1]-i[-1])<=tol_or_base:
                        indx = el
                        if ind[1]==el_lbl[1] and ind[2]==el_lbl[2]:
                            new_ind.append(1)
                        ind = el_lbl
                        size_b = size_b2
                        check_ind.append(1)
                        break
            if len(check_ind)==0 and len(vals_vox_next)>=2 and len(np.unique(lbl_nxt))>=2:
                ind = data_lbl[np.where((data_lbl[:,0]==i[0]+1) & (data_lbl[:,-2]==lbl_nxt2[-2,0]))][0]
                indx = np.where(c[:,-2]==ind[-2])[0][0]
                dist = np.sqrt((i[1]-ind[1])**2+(i[2]-ind[2])**2)
        sizes = float(size_a+size_b)/2.0
        tol_sp = int(sizes/3.7) if sizes>100 else int(sizes/2.5)
        tol_sp = tol_sp_base if tol_sp < tol_sp_base else tol_sp
        i, ind = correct_flat_ang(i, ind, tol_or_base)
        tol_or = tol_or_base if min(size_a, size_b)>small_segm[0] else tol_or_base+(tol_or_base/small_segm[1])
        ind_check = data_lbl_2[np.where((data_lbl_2[:,0]==ind[0]) & (data_lbl_2[:,1]==ind[1]) & (data_lbl_2[:,2]==ind[2]))][0]
        if ind_check[-2]!=ind[-2]:
            n, dist = check_if_continuous (i, ind, lbl_coupled)
            size_b = cp.copy(size_a)
        else:
            n=-1
        dist = -1 if len(new_ind)>0 else dist
        if float(min(size_a, size_b))>=float(max(size_a, size_b))/3 and dist<=tol_sp and abs(i[-1]-ind[-1])<=tol_or and n<0: 
            lbl_coupled[lbl_coupled==ind[-2]]=i[-2]
            p = np.where((data_lbl[:,-2]==ind[-2]) & (data_lbl[:,0]==i[0]+1))[0]
            u = data_lbl[p][0].copy(); u[-2]=i[-2]
            data_lbl[p]=u
        else:
            not_connected+=1
    print ("Not connected elements:\t{} of {} ({}% connected)".format(not_connected, all_elements, 100-int(not_connected*100/all_elements)))
    return data_lbl, lbl_coupled

def check_if_continuous (i, ind, lbl_coupled):
    """It returns a negative value if the label is continuous"""
    coords_i = np.column_stack(np.where(lbl_coupled[int(i[0])]==i[-2]))
    coords_ind = np.column_stack(np.where(lbl_coupled[int(i[0])]==ind[-2]))
    c_i_y = int((np.amax(coords_i[:,0])-np.amin(coords_i[:,0]))/2)
    c_i_x = int((np.amax(coords_i[:,1])-np.amin(coords_i[:,1]))/2)
    c_ind_y = int((np.amax(coords_ind[:,0])-np.amin(coords_ind[:,0]))/2)
    c_ind_x = int((np.amax(coords_ind[:,1])-np.amin(coords_ind[:,1]))/2)
    or_ind = svd_or(coords_ind[:,1],coords_ind[:,0])
    aver_or = (i[-1]+or_ind)/2
    n=[]
    distance = []
    if 135<aver_or<225:
        coords_i = coords_i[coords_i[:,0].argsort()]
        coords_ind = coords_ind[coords_ind[:,0].argsort()]
        if c_i_y<=c_ind_y:
            n.append(coords_i[-1,0]-coords_ind[0,0])
            distance.append(np.sqrt((coords_i[-1,0]-coords_ind[0,0])**2+(coords_i[-1,1]-coords_ind[0,1])**2))
        else:
            n.append(coords_ind[-1,0]-coords_i[0,0])
            distance.append(np.sqrt((coords_i[0,0]-coords_ind[-1,0])**2+(coords_i[0,1]-coords_ind[-1,1])**2))
    else:
        coords_i = coords_i[coords_i[:,1].argsort()]
        coords_ind = coords_ind[coords_ind[:,1].argsort()]
        if c_i_x<=c_ind_x:
            n.append(coords_i[-1,1]-coords_ind[0,1])
            distance.append(np.sqrt((coords_i[-1,0]-coords_ind[0,0])**2+(coords_i[-1,1]-coords_ind[0,1])**2))
        else:
            n.append(coords_ind[-1,1]-coords_i[0,1])
            distance.append(np.sqrt((coords_i[0,0]-coords_ind[-1,0])**2+(coords_i[0,1]-coords_ind[-1,1])**2))
    n = n[0] if n else n
    distance = distance[0] if distance else distance
    return n, distance
#%% Set path and parameters
path  = r"put_path_here"
# 2D Parameters for pairing
tol_or = 45 #Orientation tolerance (in degrees)
loc_tol_or = 45 #Local orientation tolerance (in degrees)
vx = 5 #Window size
small_segm = [5, 1.5] #small_segm[1] is the value of how much tol_or is increased for the labels smaller than small_segm[0]
tol_dist_i = 20 #Maximum distance for the intersection in the previous image for 3D continuity
# 3D Parameters for pairing
tol_sp_base = 15 #Distance tolerance (in pixels)
tol_or_base = 35 #Orientation tolerance (in degrees)
#%% Open Images and run the analysis
import timeit
st = timeit.default_timer() #Start timing the function
img = import_image_stack (path, n=1) #n is the pluing used to open the image
# Skeletonize image
lbl = skeletonize (img)
del img
# Find Interconnections coordinates and return image without interconnections
lbl, int_3 = find_interconnections(lbl)
# Remove small spots (artifact from skeletonization) less than the value indicated in the function
lbl, int_3 = remove_spots(lbl, 2, int_3)
# Acquire centroid coordinates and orientation for each label
data_lbl, lbl = data_labels(lbl)
lbl_c = lbl.copy ()
lbl, data_lbl, data_lbl_c = couple_at_intersections(lbl, vx, data_lbl, int_3, tol_or, small_segm, tol_dist_i, loc_tol_or)
# Re-Calculate centroids for the new segments
del data_lbl
data_lbl, lbl_coupled = data_labels_coupled (lbl)
# Pair in 3D using centroids
_, lbl_coupled = connect_labels_3d (data_lbl, data_lbl_c, lbl_coupled, lbl_c, tol_sp_base, tol_or_base, small_segm, int_3, vx)
#Save the results
os.chdir(path)
io.imsave("0fsf_result.tif", lbl_coupled)
del lbl_coupled
# END
lapsed_1 = timeit.default_timer(); print ("\nThe function took {}s".format(round(timeit.default_timer() - st), 2))