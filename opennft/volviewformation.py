from multiprocessing import shared_memory

import numpy as np
import multiprocessing as mp
import nibabel as nib
import cv2
import pydicom
from scipy import linalg
from rtspm import spm_imatrix, spm_matrix, spm_slice_vol
from opennft.utils import img2d_vol3d, vol3d_img2d, get_mosaic_dim
from opennft.mapimagewidget import MapImageThresholdsCalculator, RgbaMapImage, Thresholds
from opennft.config import config as con
from loguru import logger


class VolViewFormation(mp.Process):

    def __init__(self, service_data):
        mp.Process.__init__(self)
        self.str_param = dict([])

        self.exchange_data = service_data

        self.thr_calculator = MapImageThresholdsCalculator(no_value=0.0)
        self.pos_image = RgbaMapImage(colormap='hot', no_value=0.0)
        self.neg_image = RgbaMapImage(colormap='Blues_r', no_value=0.0)

        self.mat_epi = self.exchange_data["vol_mat"]
        self.dim = self.exchange_data["vol_dim"]

        self.xdim, self.ydim, self.img2d_dimx, self.img2d_dimy = get_mosaic_dim(self.dim)

        # if not (self.exchange_data["anat_volume"] is None):
        #     anat_name = self.exchange_data["anat_volume"]
        #     anat_data = nib.load(anat_name, mmap=False)
        #     self.anat_volume = np.array(anat_data.get_fdata(), order="F")
        #     self.mat_anat = anat_data.affine

        # ROIs
        if self.exchange_data["is_ROI"]:
            self.ROI_vols = np.array(self.exchange_data["ROI_vols"], order='F')
            self.ROI_mats = np.array(self.exchange_data["ROI_mats"], order='F')
        else:
            self.ROI_vols = []
            self.ROI_mats = []

        self.prepare_orth_view(self.mat_epi, self.dim)

        bb = self.str_param['bb']
        dims = np.squeeze(np.round(np.diff(bb, axis=0).T + 1))
        self.exchange_data["proj_dims"] = dims.astype(np.int32)

    def init_shmem(self):

        self.mosaic_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[1])
        mosaic_array = np.zeros((self.img2d_dimx, self.img2d_dimy, 1), dtype=np.float32)
        overlay_dim = (self.img2d_dimx, self.img2d_dimy, 4)
        self.mosaic_template = np.ndarray(shape=mosaic_array.shape,
                                          dtype=mosaic_array.dtype,
                                          buffer=self.mosaic_shmem.buf)
        self.pos_mosaic_overlay = np.ndarray(shape=overlay_dim,
                                              dtype=mosaic_array.dtype,
                                              offset=mosaic_array.nbytes+1,
                                              buffer=self.mosaic_shmem.buf)
        self.neg_mosaic_overlay = np.ndarray(shape=overlay_dim,
                                              dtype=mosaic_array.dtype,
                                              offset=mosaic_array.nbytes*4+1,
                                              buffer=self.mosaic_shmem.buf)

        self.epi_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[2])
        self.epi_volume = np.ndarray(shape=self.dim,
                                     dtype=np.float32,
                                     buffer=self.epi_shmem.buf,
                                     order="F")

        stat_dim = 2, self.dim[0], self.dim[1], self.dim[2]
        self.stat_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[3])
        self.stat_volume = np.ndarray(shape=stat_dim, dtype=np.float32, buffer=self.stat_shmem.buf, order="F")

        dims = self.exchange_data["proj_dims"]
        proj_array = np.zeros((dims[0],dims[1], 1), dtype=np.float32)
        self.proj_t_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[5])
        self.proj_t = np.ndarray(shape=(dims[0],dims[1], 1),
                                 dtype=np.float32,
                                 buffer=self.proj_t_shmem.buf,
                                 order="F")

        self.proj_t_pos_overlay = np.ndarray(shape=(dims[0],dims[1], 4),
                                             dtype=np.float32,
                                             offset=proj_array.nbytes+1,
                                             buffer=self.proj_t_shmem.buf,
                                             order="F")

        self.proj_t_neg_overlay = np.ndarray(shape=(dims[0],dims[1], 4),
                                             dtype=np.float32,
                                             offset=proj_array.nbytes*4+1,
                                             buffer=self.proj_t_shmem.buf,
                                             order="F")

        proj_array = np.zeros((dims[0],dims[2], 1), dtype=np.float32)
        self.proj_c_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[6])
        self.proj_c = np.ndarray(shape=(dims[0],dims[2], 1),
                                 dtype=np.float32,
                                 buffer=self.proj_c_shmem.buf,
                                 order="F")

        self.proj_c_pos_overlay = np.ndarray(shape=(dims[0],dims[2], 4),
                                             dtype=np.float32,
                                             offset=proj_array.nbytes+1,
                                             buffer=self.proj_c_shmem.buf,
                                             order="F")

        self.proj_c_neg_overlay = np.ndarray(shape=(dims[0],dims[2], 4),
                                             dtype=np.float32,
                                             offset=proj_array.nbytes*4+1,
                                             buffer=self.proj_c_shmem.buf,
                                             order="F")

        proj_array = np.zeros((dims[1],dims[2], 1), dtype=np.float32)
        self.proj_s_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[7])
        self.proj_s = np.ndarray(shape=(dims[1],dims[2], 1),
                                 dtype=np.float32,
                                 buffer=self.proj_s_shmem.buf,
                                 order="F")

        self.proj_s_pos_overlay = np.ndarray(shape=(dims[1],dims[2], 4),
                                             dtype=np.float32,
                                             offset=proj_array.nbytes+1,
                                             buffer=self.proj_s_shmem.buf,
                                             order="F")

        self.proj_s_neg_overlay = np.ndarray(shape=(dims[1],dims[2], 4),
                                             dtype=np.float32,
                                             offset=proj_array.nbytes*4+1,
                                             buffer=self.proj_s_shmem.buf,
                                             order="F")


    def run(self):

        self.init_shmem()

        np.seterr(divide='ignore', invalid='ignore')

        while not self.exchange_data["is_stopped"]:

            if self.exchange_data["ready_to_form"]:

                if self.exchange_data["view_mode"] == 'mosaic':

                    img_vol = self.epi_volume
                    max_vol = np.max(img_vol)
                    min_vol = np.min(img_vol)
                    img_vol = (img_vol - min_vol) / (max_vol - min_vol)
                    self.mosaic_template[:,:,:] = np.expand_dims(vol3d_img2d(img_vol, self.xdim, self.ydim,
                                                                 self.img2d_dimx, self.img2d_dimy, self.dim), 2)

                    self.exchange_data["done_mosaic_templ"] = True

                    if self.exchange_data["overlay_ready"]:
                        if self.exchange_data["is_rtqa"]:
                            overlay_vol = self.exchange_data["rtQA_volume"]
                            overlay_img = vol3d_img2d(overlay_vol, self.xdim, self.ydim,
                                                      self.img2d_dimx, self.img2d_dimy, self.dim)
                            overlay_img = (overlay_img / np.max(overlay_img)) * 255
                        else:
                            overlay_vol = self.stat_volume[0,:,:,:].squeeze()
                            overlay_img = vol3d_img2d(overlay_vol, self.xdim, self.ydim,
                                                      self.img2d_dimx, self.img2d_dimy, self.dim)
                            overlay_img = (overlay_img / np.max(overlay_img)) * 255

                            if self.exchange_data["is_neg"]:
                                neg_overlay_vol = self.stat_volume[1,:,:,:].squeeze()
                                neg_overlay_img = vol3d_img2d(neg_overlay_vol, self.xdim, self.ydim,
                                                              self.img2d_dimx, self.img2d_dimy, self.dim)
                                neg_mosaic_overlay = (neg_overlay_img / np.max(neg_overlay_img)) * 255

                        if self.exchange_data["auto_thr_pos"]:
                            pos_thr = self.thr_calculator(overlay_img)
                            if pos_thr.lower < 0:
                                pos_thr = Thresholds(0, pos_thr.upper)
                            self.exchange_data["pos_thresholds"] = pos_thr
                        else:
                            pos_thr = self.exchange_data["pos_thresholds"]
                        self.pos_mosaic_overlay[:,:,:] = self.pos_image(overlay_img, pos_thr, 1.0)
                        print(self.pos_mosaic_overlay.mean())

                        if self.exchange_data["is_neg"]:
                            if self.exchange_data["auto_thr_neg"]:
                                neg_thr = self.thr_calculator(neg_overlay_img)
                                if neg_thr.lower < 0:
                                    neg_thr = Thresholds(0, neg_thr.upper)
                                self.exchange_data["neg_thresholds"] = neg_thr
                            else:
                                neg_thr = self.exchange_data["neg_thresholds"]
                            self.neg_mosaic_overlay[:,:,:] = self.neg_image(neg_overlay_img, neg_thr, 1.0)

                        self.exchange_data["done_mosaic_overlay"] = True


                else:

                    flags = [self.exchange_data["bg_type"], self.exchange_data["is_neg"], self.exchange_data["is_ROI"]]

                    # background
                    if flags[0] == "bgEPI":
                        back_volume = self.epi_volume
                        mat = self.mat_epi
                    else:
                        back_volume = self.anat_volume
                        mat = self.mat_anat

                    ROI_vols = []
                    ROI_mats = []

                    cursor_pos = self.exchange_data["cursor_pos"]
                    flags_planes = self.exchange_data["flags_planes"]

                    proj = np.nonzero(flags_planes)

                    new_coord = np.array([[0, 0],[0, 0],[0, 0]])
                    new_coord[proj,:] = cursor_pos

                    self.str_param['centre'] = self.findcent(new_coord, flags_planes)
                    # Display modes: [Background + Stat + ROIs, Background + Stat, Background + ROIs]
                    self.str_param['mode_displ'] = [1, 0, 0]

                    [proj_t, proj_c, proj_s,
                     overlay_t, overlay_c, overlay_s,
                     neg_overlay_t, neg_overlay_c, neg_overlay_s,
                     ROI_t, ROI_c, ROI_s
                    ] = self.update_orth_view(back_volume, mat,
                                              self.stat_volume[0,:,:,:].squeeze(),
                                              self.stat_volume[1,:,:,:].squeeze(),
                                              ROI_vols, ROI_mats, flags)

                    self.proj_t[:, :, :] = np.expand_dims(proj_t, 2)
                    self.proj_c[:, :, :] = np.expand_dims(proj_c, 2)
                    self.proj_s[:, :, :] = np.expand_dims(proj_s, 2)

                    pos_maps_values = np.array(overlay_t.ravel(), dtype=np.uint8)
                    pos_maps_values = np.append(pos_maps_values, overlay_c.ravel())
                    pos_maps_values = np.append(pos_maps_values, overlay_s.ravel())
                    if self.exchange_data["auto_thr_pos"]:
                        pos_thr = self.thr_calculator(pos_maps_values)
                        if (not pos_thr is None) and pos_thr.lower < 0:
                            pos_thr = Thresholds(0, pos_thr.upper)
                        self.exchange_data["pos_thresholds"] = pos_thr
                    else:
                        pos_thr = self.exchange_data["pos_thresholds"]
                    self.proj_t_pos_overlay[:, :, :] = self.pos_image(overlay_t, pos_thr, 1.0)
                    self.proj_c_pos_overlay[:, :, :] = self.pos_image(overlay_c, pos_thr, 1.0)
                    self.proj_s_pos_overlay[:, :, :] = self.pos_image(overlay_s, pos_thr, 1.0)

                    if self.exchange_data["is_neg"]:
                        neg_maps_values = np.array(neg_overlay_t.ravel(), dtype=np.uint8)
                        neg_maps_values = np.append(neg_maps_values, neg_overlay_c.ravel())
                        neg_maps_values = np.append(neg_maps_values, neg_overlay_s.ravel())
                        if self.exchange_data["auto_thr_neg"]:
                            neg_thr = self.thr_calculator(neg_maps_values)
                            if (not neg_thr is None) and neg_thr.lower < 0:
                                neg_thr = Thresholds(0, neg_thr.upper)
                            self.exchange_data["neg_thresholds"] = neg_thr
                        else:
                            neg_thr = self.exchange_data["neg_thresholds"]
                        self.proj_t_neg_overlay[:, :, :] = self.neg_image(neg_overlay_t, neg_thr, 1.0)
                        self.proj_c_neg_overlay[:, :, :] = self.neg_image(neg_overlay_c, neg_thr, 1.0)
                        self.proj_s_neg_overlay[:, :, :] = self.neg_image(neg_overlay_s, neg_thr, 1.0)

                    self.exchange_data["done_orth"] = True

                self.exchange_data["ready_to_form"] = False

        self.mosaic_shmem.close()
        self.epi_shmem.close()
        self.stat_shmem.close()
        self.proj_t_shmem.close()
        self.proj_c_shmem.close()
        self.proj_s_shmem.close()

    def prepare_orth_view(self, mat, dim):
        # set structure for Display and draw a first overlay
        self.str_param = {'n': 0, 'bb': [], 'space': np.eye(4, 4), 'centre': np.zeros((1, 3)), 'mode': 1,
                     'area': np.array([0, 0, 1, 1]), 'premul': np.eye(4, 4), 'hld': 1, 'mode_displ': np.zeros((1, 3))}

        temp = np.array([0, 0, 0, 0, np.pi, -np.pi / 2])
        self.str_param['space'] = spm_matrix(temp) @ self.str_param['space']

        # get bounding box and resolution
        if len(self.str_param['bb']) == 0:
            self.str_param['max_bb'] = self.max_bb(mat, dim, self.str_param['space'], self.str_param['premul'])
            self.str_param['bb'] = self.str_param['max_bb']

        self.str_param['space'], self.str_param['bb'] = self.resolution(mat, self.str_param['space'], self.str_param['bb'])

        # Draw at initial location, center of bounding box
        temp = np.vstack((self.str_param['max_bb'].T, [1, 1]))
        mmcentre = np.mean(self.str_param['space'] @ temp, 1)
        self.str_param['centre'] = mmcentre[0:3]
        # Display modes: [Background + Stat + ROIs, Background + Stat, Background + ROIs]
        self.str_param['mode_displ'] = np.array([0, 0, 1])

    def update_orth_view(self, vol, mat, overlay_vol, neg_overlay_vol, ROI_vols, ROI_mats, flags):

        bb = self.str_param['bb']
        dims = np.squeeze(np.round(np.diff(bb, axis=0).T + 1))
        _is = np.linalg.inv(self.str_param['space'])
        cent = _is[0:3, 0:3] @ self.str_param['centre'] + _is[0:3, 3]

        m = np.array(np.linalg.solve(self.str_param['space'], self.str_param['premul']) @ mat, order='F')
        tm0 = np.array([
            [1, 0, 0, -bb[0, 0] + 1],
            [0, 1, 0, -bb[0, 1] + 1],
            [0, 0, 1, -cent[2]],
            [0, 0, 0, 1],

        ])
        td = np.array([dims[0], dims[1]], dtype=int, order='F')

        cm0 = np.array([
            [1, 0, 0, -bb[0, 0] + 1],
            [0, 0, 1, -bb[0, 2] + 1],
            [0, 1, 0, -cent[1]],
            [0, 0, 0, 1],
        ])
        cd = np.array([dims[0], dims[2]], dtype=int, order='F')

        if self.str_param['mode'] == 0:
            sm0 = np.array([
                [0, 0, 1, -bb[0, 2] + 1],
                [0, 1, 0, -bb[0, 1] + 1],
                [1, 0, 0, -cent[0]],
                [0, 0, 0, 1],
            ])
            sd = np.array([dims[2], dims[1]], dtype=int, order='F')
        else:
            sm0 = np.array([
                [0, -1, 0, +bb[1, 1] + 1],
                [0, 0, 1, -bb[0, 2] + 1],
                [1, 0, 0, -cent[0]],
                [0, 0, 0, 1],
            ])
            sd = np.array([dims[1], dims[2]], dtype=int, order='F')

        coord_param = {'tm0': tm0, 'cm0': cm0, 'sm0': sm0, 'td': td, 'cd': cd, 'sd': sd}

        back_imgt, back_imgc, back_imgs = self.get_orth_vol(coord_param, vol, m)

        back_imgt = np.nan_to_num(back_imgt)
        back_imgt[back_imgt < 0] = 0

        back_imgc = np.nan_to_num(back_imgc)
        back_imgc[back_imgc < 0] = 0

        back_imgs = np.nan_to_num(back_imgs)
        back_imgs[back_imgs < 0] = 0

        back_imgt = ((back_imgt / np.max(back_imgt)) * 255)
        back_imgc = ((back_imgc / np.max(back_imgc)) * 255)
        back_imgs = ((back_imgs / np.max(back_imgs)) * 255)

        if flags[0] != "bgEPI":
            m = np.array(np.linalg.solve(self.str_param['space'], self.str_param['premul']) @ self.mat_epi, order='F')

        overlay_imgt, overlay_imgc, overlay_imgs = self.get_orth_vol(coord_param, overlay_vol, m)
        overlay_imgt = np.nan_to_num(overlay_imgt)
        overlay_imgc = np.nan_to_num(overlay_imgc)
        overlay_imgs = np.nan_to_num(overlay_imgs)
        overlay_imgt = ((overlay_imgt / np.max(overlay_imgt)) * 255)
        overlay_imgc = ((overlay_imgc / np.max(overlay_imgc)) * 255)
        overlay_imgs = ((overlay_imgs / np.max(overlay_imgs)) * 255)

        if flags[1]:
            neg_overlay_imgt, neg_overlay_imgc, neg_overlay_imgs = self.get_orth_vol(coord_param, neg_overlay_vol, m)
            neg_overlay_imgt = np.nan_to_num(neg_overlay_imgt)
            neg_overlay_imgc = np.nan_to_num(neg_overlay_imgc)
            neg_overlay_imgs = np.nan_to_num(neg_overlay_imgs)
            neg_overlay_imgt = ((neg_overlay_imgt / np.max(neg_overlay_imgt)) * 255)
            neg_overlay_imgc = ((neg_overlay_imgc / np.max(neg_overlay_imgc)) * 255)
            neg_overlay_imgs = ((neg_overlay_imgs / np.max(neg_overlay_imgs)) * 255)

        else:
            neg_overlay_imgt = None
            neg_overlay_imgc = None
            neg_overlay_imgs = None

        nrROIs = self.exchange_data["nr_rois"]
        ROI_t = [None]*nrROIs
        ROI_c = [None]*nrROIs
        ROI_s = [None]*nrROIs

        if bool(self.str_param["mode_displ"]) and flags[2]:

            for j in range(nrROIs):

                vol = np.array(np.squeeze(ROI_vols[j,:,:,:]), order='F')
                mat = np.array(np.squeeze(ROI_mats[j,:,:]), order='F')
                m = np.array(np.linalg.solve(self.str_param['space'], self.str_param['premul']) @ mat, order='F')
                temp_t, temp_c, temp_s = self.get_orth_vol(coord_param, vol, m)

                ROI_t[j] = self.roi_boundaries(temp_t)
                ROI_c[j] = self.roi_boundaries(temp_c)
                ROI_s[j] = self.roi_boundaries(temp_s)

        return back_imgt, back_imgc, back_imgs, overlay_imgt, overlay_imgc, overlay_imgs, neg_overlay_imgt, neg_overlay_imgc, neg_overlay_imgs, ROI_t, ROI_c, ROI_s

    def roi_boundaries(self, roi):
        roi[np.isnan(roi)] = 0
        contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            boundaries = [None] * len(contours)
            for i in range(len(contours)):
                boundaries[i] = contours[i].squeeze()
        else:
            boundaries = np.array([])

        return boundaries

    def get_orth_vol(self, coord_param, vol, m):
        temp = np.array([0, np.nan], order='F')

        mat_t = np.array(linalg.inv(coord_param['tm0'] @ m), order='F')
        imgt = np.zeros((coord_param['td'][0], coord_param['td'][1]), order='F')
        spm_slice_vol(vol, imgt, mat_t, temp)
        imgt = imgt

        mat_c = np.array(linalg.inv(coord_param['cm0'] @ m), order='F')
        imgc = np.zeros((coord_param['cd'][0], coord_param['cd'][1]), order='F')
        spm_slice_vol(vol, imgc, mat_c, temp)
        imgc = imgc

        mat_s = np.array(linalg.inv(coord_param['sm0'] @ m), order='F')
        imgs = np.zeros((coord_param['sd'][0], coord_param['sd'][1]), order='F')
        spm_slice_vol(vol, imgs, mat_s, temp)
        imgs = imgs

        return imgt, imgc, imgs

    def findcent(self, coord_loc, flags_planes):

        centre = np.array([])
        cent = np.array([])
        cp = np.array([])

        for j in range(3):
            if flags_planes[j]: # new coordinates on Transverse
                cp = np.array(coord_loc[j,:],ndmin=2)
            if cp.size > 0:
                cp = cp[0,0:2]
                _is = np.linalg.inv(self.str_param['space'])
                cent = _is[0:3, 0:3] @ self.str_param['centre'] + _is[0:3, 3]
                if j == 0: # click was on Transverse: s and t need to change
                    cent[0] = cp[0] + self.str_param['bb'][0,0] - 1
                    cent[1] = cp[1] + self.str_param['bb'][0,1] - 1
                elif j == 1: # click was on Saggital: t and c need to change
                    cent[0] = cp[0] + self.str_param['bb'][0,0] - 1
                    cent[2] = cp[1] + self.str_param['bb'][0,2] - 1
                elif j == 2: # click was on Coronal: t and s need to change
                    if self.str_param['mode'] == 0:
                        cent[2] = cp[0] + self.str_param['bb'][0,2] - 1
                        cent[1] = cp[1] + self.str_param['bb'][0,1] - 1
                    else:
                        cent[1] = cp[0] - self.str_param['bb'][1,1] - 1
                        cent[2] = cp[1] + self.str_param['bb'][0,2] - 1
                break

        if cent.size>0:
            centre = self.str_param['space'][0:3,0:3] @ cent[:] + self.str_param['space'][0:3,3]

        return centre

    def max_bb(self, mat, dim, space, premul):

        mn = np.array([np.inf] * 3, ndmin=2)
        mx = -mn
        premul = np.linalg.solve(space, premul)
        bb, vx = self.get_bbox(mat, dim, premul)
        mx = np.vstack((bb, mx)).max(0)
        mn = np.vstack((bb, mx)).min(0)
        bb = np.vstack((mn, mx))

        return bb

    def get_bbox(self, mat, dim, premul):
        p = spm_imatrix(mat)
        vx = p[6:9]

        corners = np.array([
            [1, 1, 1, 1],
            [1, 1, dim[2], 1],
            [1, dim[1], 1, 1],
            [1, dim[1], dim[2], 1],
            [dim[0], 1, 1, 1],
            [dim[0], 1, dim[2], 1],
            [dim[0], dim[1], 1, 1],
            [dim[0], dim[1], dim[2], 1],

        ]).T

        xyz = mat[0:3, :] @ corners

        xyz = premul[0:3, :] @ np.vstack((xyz, np.ones((1, xyz.shape[1]))))

        bb = np.array([
            np.min(xyz, axis=1).T,
            np.max(xyz, axis=1).T
        ])

        return bb, vx

    def resolution(self, mat, space, bb):
        res_default = 1

        temp = (np.sum((mat[0:3, 0:3]) ** 2, axis=0) ** .5)
        res = np.min(np.hstack((res_default, temp)))

        u, s, v = np.linalg.svd(space[0:3, 0:3])
        temp = np.mean(s)
        res = res / temp

        mat = np.diag([res, res, res, 1])

        space = space @ mat
        bb = bb / res

        return space, bb
