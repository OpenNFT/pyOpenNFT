# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib
import pydicom

from pathlib import Path
from loguru import logger
from scipy.io import savemat

from opennft.mrvol import MrVol
from opennft.mrroi import MrROI
from opennft.mrtimeseries import MrTimeSeries
from opennft.iglm_vol import iglm_vol
from opennft.utils import get_mosaic_dim, img2d_vol3d, ar_regr, zscore, get_mat
from opennft.config import config as con


# --------------------------------------------------------------------------
class NftSession():
    """Session contains data like P structure
    """

    # --------------------------------------------------------------------------
    def __init__(self, config):

        self.config = config
        self.reference_vol = MrVol()  # mc_template
        self.dim = (config.matrix_size_x, config.matrix_size_y, config.slices_nr)
        self.vect_end_cond = np.ones((self.config.volumes_nr - self.config.skip_vol_nr, 1))
        self.first_nf_inds = []
        self.pos_contrast = []
        self.neg_contrast = []
        self.nr_vols = config.volumes_nr - config.skip_vol_nr
        self.nr_rois = 0
        self.rois = []
        self.roi_names = []
        self.xdim_img_count = 0  # number of image in mosaic per horizontal
        self.ydim_img_count = 0  # number of image in mosaic per vertical
        self.img2d_dimx = 0  # mosaic image size X
        self.img2d_dimy = 0  # mosaic image size Y
        self.spm = None
        self.dvars_scale = con.default_dvars_threshold

    def setup(self, mc_templ_path):

        self.reference_vol.load_vol(mc_templ_path, "nii")
        self.xdim_img_count, self.ydim_img_count, self.img2d_dimx, self.img2d_dimy = get_mosaic_dim(
            self.reference_vol.dim)

    def setup_auto_rtqa(self, mc_template_file):

        dcm = pydicom.dcmread(mc_template_file, force=True)

        if not (hasattr(dcm, 'ImagePositionPatient') and hasattr(dcm, 'ImageOrientationPatient')):
            logger.error("DICOM template has no ImagePositionPatient and ImageOrientationPatient and "
                         "could not be used as EPI template\nPlease, check DICOM export or use NII EPI template\n")
            return False

        self.reference_vol.dim = np.array(self.dim)
        self.reference_vol.mat = get_mat(self.dim, dcm)
        self.xdim_img_count, self.ydim_img_count, self.img2d_dimx, self.img2d_dimy = get_mosaic_dim(
            self.reference_vol.dim)
        self.reference_vol.volume = np.array(img2d_vol3d(dcm.pixel_array, self.xdim_img_count,
                                                         self.ydim_img_count,
                                                         self.reference_vol.dim), order='F')

        return True

    def set_protocol(self, simulation_protocol):

        if not con.auto_rtqa:
            conditions = simulation_protocol["ConditionIndex"]
            cond_length = len(conditions)

            self.prot_names = []
            self.offsets = []
            self.prot_cond = [None] * (cond_length + 1)
            inc = 2
            for i in range(cond_length):
                self.prot_names.append(conditions[i]["ConditionName"])

                # TODO: check if baseline field already exists in protocol
                offsets = np.array(conditions[i]["OnOffsets"])

                for j in range(len(offsets)):
                    self.vect_end_cond[offsets[j][0] - 1:offsets[j][1]] = inc
                    if self.prot_cond[inc - 1] is None:
                        self.prot_cond[inc - 1] = np.array(range(offsets[j][0], offsets[j][1] + 1))
                    else:
                        self.prot_cond[inc - 1] = np.vstack(
                            (self.prot_cond[inc - 1], np.array(range(offsets[j][0], offsets[j][1] + 1))))

                self.first_nf_inds.append(offsets[:, 0] - 1)

                self.offsets.append(offsets)

                inc = inc + 1

            # Implicit baseline
            bas_inds = np.where(self.vect_end_cond == 1)[0] + 1
            bas_offsets = []
            for i in range(len(self.offsets[0])):
                if i == 0:
                    inds = bas_inds[np.where(bas_inds < self.offsets[0][i][0])[0]]
                else:
                    inds = bas_inds[np.where(np.logical_and(bas_inds > self.offsets[0][i - 1][1],
                                                            bas_inds < self.offsets[0][i][0]))[0]]

                bas_offsets.append(inds)

            inds = bas_inds[np.where(bas_inds > self.offsets[0][-1][1])[0]]
            if len(inds) > 0:
                bas_offsets.append(inds)
            # bas_offsets = np.array(bas_offsets, ndmin=2)
            self.prot_cond[0] = bas_offsets

            # Contrast and Conditions For Contrast
            if "ContrastActivation" in simulation_protocol:
                splitted_contrast = simulation_protocol["ContrastActivation"].split("*")
                self.pos_contrast = np.array(splitted_contrast[0::2], dtype=np.int32)
                self.neg_contrast = -self.pos_contrast

        else:

            self.pos_contrast = np.array([1, 1, 1, 1, 1, 1], ndmin=2, dtype=np.int32).T
            self.neg_contrast = -self.pos_contrast

    def select_rois(self):

        for roi_file in Path(self.config.roi_files_dir).iterdir():
            if roi_file.is_file():
                self.nr_rois += 1
                new_roi = MrROI()
                new_roi.load_roi(roi_file.absolute())
                self.rois.append(new_roi)
                self.roi_names.append(new_roi.name)

        if self.config.type == "SVM":
            for weight_file, ind_roi in zip(Path(self.config.weights_file_name).iterdir(), range(self.nr_rois)):
                self.rois[ind_roi].load_weights(weight_file)

    def wb_roi_init(self):

        if con.use_rtqa:
            wb_roi = MrROI()
            self.dvars_scale = wb_roi.load_whole_brain_mask(self.reference_vol.clone())
            self.rois.append(wb_roi)
            self.roi_names.append(wb_roi.name)

            if not con.auto_rtqa or con.use_epi_template:
                self.nr_rois += 1


# --------------------------------------------------------------------------
class NftIteration:
    """Iteration contains data like main_loop_data
    """

    # --------------------------------------------------------------------------
    def __init__(self, session):
        self.pre_iter = -1
        self.iter_number = 0
        self.iter_norm_number = 0
        self.nr_blocks_in_sliding_window = 100
        self.mr_vol = MrVol()
        self.mr_time_series = MrTimeSeries(session.nr_rois)
        self.session = session
        self.bas_func = []
        self.sig_prproc_glm_design = []
        self.lin_regr = []

        if not con.auto_rtqa:
            if con.iglm_ar1:
                self.bas_func = ar_regr(con.a_ar1, session.spm["xX_x"][:, 0:-1])
            else:
                self.bas_func = session.spm["xX_x"][:, 0:-2]

            self.nr_bas_func = len(self.bas_func[0])

        else:
            self.nr_bas_func = 6

        if not con.is_regr_iglm:
            if not con.auto_rtqa:
                self.nr_bas_fct_regr = 1
            else:
                self.nr_bas_fct_regr = 6

        else:
            nr_high_pass_regr = len(session.spm["xX_K"]["x0"][0])
            is_high_pass = con.is_high_pass
            is_motion_regr = con.is_motion_regr
            is_lin_regr = con.is_lin_regr

            if not con.auto_rtqa:
                nr_mot_regr = 6
                if is_high_pass and is_motion_regr and is_lin_regr:
                    self.nr_bas_fct_regr = nr_mot_regr + nr_high_pass_regr + 2
                    # adding 6 head motion, linear, high-pass filter, and constant regressors
                elif not is_high_pass and is_motion_regr and is_lin_regr:
                    self.nr_bas_fct_regr = nr_mot_regr + 2
                    # adding 6 head motion, linear, and constant regressors
                elif is_high_pass and not is_motion_regr and is_lin_regr:
                    self.nr_bas_fct_regr = nr_high_pass_regr + 2
                    # dding high-pass filter, linear, and constant regressors
                elif is_high_pass and not is_motion_regr and not is_lin_regr:
                    self.nr_bas_fct_regr = nr_high_pass_regr + 1
                    # adding high-pass filter, and constant regressors
                elif not is_high_pass and not is_motion_regr and is_lin_regr:
                    self.nr_bas_fct_regr = 2  # adding linear, and constant regressors

            else:
                if is_high_pass and is_lin_regr:
                    self.nr_bas_fct_regr = nr_high_pass_regr + 2
                elif not is_high_pass and is_lin_regr:
                    self.nr_bas_fct_regr = 2
                elif is_high_pass and not is_lin_regr:
                    self.nr_bas_fct_regr = nr_high_pass_regr + 1

        # realignment parameters
        self.a0 = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.deg = []
        self.b = []

        n = self.nr_bas_fct_regr + self.nr_bas_func
        nr_vox_in_vol = self.session.dim[0] * self.session.dim[1] * self.session.dim[2]
        # iGLM parameters
        self.iglm_params = {
            "p_val": .01,
            "t_contr_pos": self.session.pos_contrast,
            "t_contr_neg": self.session.neg_contrast,
            "cn": np.zeros((n, n)),
            "dn": np.zeros((nr_vox_in_vol, n)),
            "s2n": np.zeros((nr_vox_in_vol, 1)),
            "tn_pos": np.zeros((nr_vox_in_vol, 1)),
            "tn_neg": np.zeros((nr_vox_in_vol, 1)),
            "t_th": np.zeros((self.session.nr_vols, 1)),
            "dyn_t_th": 0,
            "spm_mask_th": session.spm["xM_TH"].mean() * np.ones(session.spm["xM_TH"].shape),
            "stat_map_vect": np.zeros((nr_vox_in_vol, 1)),
            "stat_map_3d_pos": np.zeros(self.session.dim),
            "stat_map_3d_neg": np.zeros(self.session.dim)
        }

    # --------------------------------------------------------------------------
    def auto_rtqa_later_reinit(self):

        self.mr_time_series = MrTimeSeries(self.session.nr_rois)

        n = self.nr_bas_fct_regr + self.nr_bas_func
        nr_vox_in_vol = self.session.dim[0] * self.session.dim[1] * self.session.dim[2]
        # iGLM parameters
        self.iglm_params = {
            "p_val": .01,
            "t_contr_pos": self.session.pos_contrast,
            "t_contr_neg": self.session.neg_contrast,
            "cn": np.zeros((n, n)),
            "dn": np.zeros((nr_vox_in_vol, n)),
            "s2n": np.zeros((nr_vox_in_vol, 1)),
            "tn_pos": np.zeros((nr_vox_in_vol, 1)),
            "tn_neg": np.zeros((nr_vox_in_vol, 1)),
            "t_th": np.zeros((self.session.nr_vols, 1)),
            "dyn_t_th": 0,
            "spm_mask_th": self.session.spm["xM_TH"].mean() * np.ones(self.session.spm["xM_TH"].shape),
            "stat_map_vect": np.zeros((nr_vox_in_vol, 1)),
            "stat_map_3d_pos": np.zeros(self.session.dim),
            "stat_map_3d_neg": np.zeros(self.session.dim)
        }

    # --------------------------------------------------------------------------
    def load_vol(self, file_name, im_type):

        self.mr_vol.load_vol(file_name, im_type)
        self.mr_vol.volume = np.array(img2d_vol3d(self.mr_vol.volume, self.session.xdim_img_count,
                                                  self.session.ydim_img_count, self.session.reference_vol.dim),
                                      order='F')

    # --------------------------------------------------------------------------
    def process_vol(self):
        r, self.a0, self.x1, self.x2, self.x3, self.deg, self.b = self.mr_vol.realign(self, self.a0, self.x1, self.x2,
                                                                                      self.x3, self.deg, self.b)
        self.mr_time_series.motion_correction_parameters(r[0]["mat"], r[1]["mat"])
        self.mr_vol.reslice(r)
        self.mr_vol.smooth()

    # --------------------------------------------------------------------------
    def process_time_series(self):
        self.mr_time_series.acquiring(self.session.config.type, self.mr_vol, self.session.rois)

        if not con.auto_rtqa:
            sl_wind = (self.session.offsets[0][0][0] - 1) * self.nr_blocks_in_sliding_window
            bas_block_length = self.session.offsets[0][0][0] - 1
        else:
            sl_wind = self.session.nr_vols
            bas_block_length = 0
        is_svm = self.session.config.type == 'SVM'

        self.mr_time_series.preprocessing(self.iter_norm_number, self.bas_func, self.lin_regr, sl_wind,
                                          self.session.vect_end_cond, bas_block_length, is_svm)

    # --------------------------------------------------------------------------
    def iglm(self):

        ind_iglm = self.iter_norm_number
        stat_ready = False

        if con.auto_rtqa:
            is_motion_regr = False
        else:
            is_motion_regr = con.is_motion_regr
        is_high_pass = con.is_high_pass
        is_lin_regr = con.is_lin_regr

        tmp_regr = np.array([])

        if con.is_regr_iglm:
            if is_high_pass and is_motion_regr and is_lin_regr:

                tmp_regr = zscore(self.mr_time_series.mc_params[:, 0:ind_iglm + 1]).T
                tmp_regr = np.hstack((tmp_regr, self.lin_regr[0:ind_iglm + 1]))
                tmp_regr = np.hstack((tmp_regr, self.session.spm["xX_K"]["x0"][0:ind_iglm + 1, :]))
                tmp_regr = np.hstack((tmp_regr, np.ones((ind_iglm + 1, 1))))
            elif not is_high_pass and is_motion_regr and is_lin_regr:

                tmp_regr = zscore(self.mr_time_series.mc_params[:, 0:ind_iglm + 1]).T
                tmp_regr = np.hstack((tmp_regr, self.lin_regr[0:ind_iglm + 1]))
                tmp_regr = np.hstack((tmp_regr, np.ones((ind_iglm + 1, 1))))

            elif is_high_pass and not is_motion_regr and is_lin_regr:

                tmp_regr = self.lin_regr[0:ind_iglm + 1]
                tmp_regr = np.hstack((tmp_regr, self.session.spm["xX_K"]["x0"][0:ind_iglm + 1, :]))
                tmp_regr = np.hstack((tmp_regr, np.ones((ind_iglm + 1, 1))))

            elif is_high_pass and not is_motion_regr and not is_lin_regr:

                tmp_regr = self.session.spm["xX_K"]["x0"][0:ind_iglm + 1, :]
                tmp_regr = np.hstack((tmp_regr, np.ones((ind_iglm + 1, 1))))

            elif not is_high_pass and not is_motion_regr and is_lin_regr:

                tmp_regr = self.lin_regr[0:ind_iglm + 1]
                tmp_regr = np.hstack((tmp_regr, np.ones((ind_iglm + 1, 1))))

        else:

            tmp_regr = np.ones((ind_iglm, 1))

        if con.iglm_ar1:
            tmp_regr = ar_regr(con.a_ar1, tmp_regr)

        if not con.auto_rtqa:
            bas_fct_regr = np.hstack((self.bas_func[0:ind_iglm + 1, :], tmp_regr))
        else:
            bas_fct_regr = np.hstack((zscore(self.mr_time_series.mc_params[:, 0:ind_iglm + 1]).T, tmp_regr))

        t_contr_pos = np.vstack((self.iglm_params["t_contr_pos"], np.zeros((self.nr_bas_fct_regr, 1))))
        t_contr_neg = np.vstack((self.iglm_params["t_contr_neg"], np.zeros((self.nr_bas_fct_regr, 1))))

        cn = self.iglm_params["cn"]
        dn = self.iglm_params["dn"]
        s2n = self.iglm_params["s2n"]
        tn = {"pos": self.iglm_params["tn_pos"], "neg": self.iglm_params["tn_neg"]}
        t_contr = {"pos": t_contr_pos, "neg": t_contr_neg}
        p_val = self.iglm_params["p_val"]
        dynt_th = self.iglm_params["dyn_t_th"]
        t_th = self.iglm_params["t_th"]
        spm_mask_th = self.iglm_params["spm_mask_th"]
        self.iglm_params["stat_map_3d_pos"] = np.zeros(self.session.reference_vol.dim)
        self.iglm_params["stat_map_3d_neg"] = np.zeros(self.session.reference_vol.dim)

        idx_act_vox, dynt_th, \
            t_th, cn, dn, sigma2n, \
            tn, neg_e2n, bn, e2n = iglm_vol(cn, dn, s2n, tn, self.mr_vol.volume.flatten(order="F"),
                                            ind_iglm + 1, self.nr_bas_func + self.nr_bas_fct_regr,
                                            t_contr, bas_fct_regr, p_val, dynt_th, t_th, spm_mask_th)

        self.iglm_params["cn"] = cn
        self.iglm_params["dn"] = dn
        self.iglm_params["s2n"] = sigma2n
        self.iglm_params["tn_pos"] = tn["pos"]
        self.iglm_params["tn_neg"] = tn["neg"]
        self.iglm_params["dyn_t_th"] = dynt_th
        self.iglm_params["t_th"] = t_th

        dim = self.session.reference_vol.dim

        if idx_act_vox["pos"].size > 0 and np.max(tn["pos"]) > 0:
            masked_stat_map_vect_pos = tn["pos"][idx_act_vox["pos"]]
            stat_map_vect = masked_stat_map_vect_pos.squeeze()
            temp_map = self.iglm_params["stat_map_3d_pos"].flatten(order='F')
            temp_map[idx_act_vox["pos"]] = stat_map_vect
            self.iglm_params["stat_map_3d_pos"] = np.reshape(temp_map, dim, order='F')
            stat_ready = True

        if idx_act_vox["neg"].size > 0 and np.max(tn["neg"]) > 0:
            masked_stat_map_vect_neg = tn["neg"][idx_act_vox["neg"]]
            stat_map_vect = masked_stat_map_vect_neg.squeeze()
            temp_map = self.iglm_params["stat_map_3d_neg"].flatten(order='F')
            temp_map[idx_act_vox["neg"]] = stat_map_vect
            self.iglm_params["stat_map_3d_neg"] = np.reshape(temp_map, dim, order='F')
            stat_ready = True

        return stat_ready

    # --------------------------------------------------------------------------
    def save_stat_vols(self, save_path):
        if len(self.mr_time_series.mc_params) == 0:
            logger.info(f"Empty data, nothing to save")
            return

        if not save_path.is_dir():
            save_path.mkdir(exist_ok=True)

        path_pos = save_path / "py_stat_pos.nii"
        path_neg = save_path / "py_stat_neg.nii"

        if not path_pos.is_file():
            path_pos.touch(exist_ok=True)

        if not path_neg.is_file():
            path_neg.touch(exist_ok=True)

        nib.save(nib.Nifti1Image(self.iglm_params["stat_map_3d_pos"], self.session.reference_vol.mat), path_pos)
        nib.save(nib.Nifti1Image(self.iglm_params["stat_map_3d_neg"], self.session.reference_vol.mat), path_neg)

    # --------------------------------------------------------------------------
    def save_time_series(self, save_path):
        if len(self.mr_time_series.mc_params) == 0:
            logger.info(f"Empty data, nothing to save")
            return

        if not save_path.is_dir():
            save_path.mkdir(exist_ok=True)

        path = save_path / "py_time_series.mat"

        if not path.is_file():
            path.touch(exist_ok=True)

        savemat(str(path), {"raw_time_series": self.mr_time_series.raw_time_series[0],
                            "raw_time_series_ar1": self.mr_time_series.raw_time_series_ar1[0],
                            "kalman_proc_time_series": self.mr_time_series.kalman_proc_time_series[0],
                            "glm_time_series": self.mr_time_series.glm_time_series[0],
                            "scale_time_series": self.mr_time_series.scale_time_series[0],
                            "x": self.mr_time_series.mc_params[0, :],
                            "y": self.mr_time_series.mc_params[1, :],
                            "z": self.mr_time_series.mc_params[2, :],
                            "pitch": self.mr_time_series.mc_params[3, :],
                            "roll": self.mr_time_series.mc_params[4, :],
                            "yaw": self.mr_time_series.mc_params[5, :],
                            })
