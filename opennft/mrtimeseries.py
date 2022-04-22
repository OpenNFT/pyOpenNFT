import numpy as np
from opennft.config import config as con

from rtspm import spm_imatrix


# --------------------------------------------------------------------------
class MrTimeSeries():

    # --------------------------------------------------------------------------
    def __init__(self, nr_rois):

        self.raw_time_series = [None] * nr_rois
        self.disp_raw_time_series = [None] * nr_rois
        self.raw_time_series_ar1 = [None] * nr_rois
        self.init_lim = [None] * nr_rois
        self.proc_time_series = [None] * nr_rois
        self.glm_time_series = [None] * nr_rois
        self.no_reg_time_series = [None] * nr_rois
        self.norm_time_series = [None] * nr_rois
        self.nr_rois = nr_rois

        # x, y, z, pitch, roll, yaw
        self.mc_params = []
        self.offset_mc = []

    def acquiring(self, type, vol, rois):

        for ind_roi in range(self.nr_rois):

            if type == "PSC":
                ts_value = np.mean(vol.volume[rois[ind_roi].voxel_index[:,0],
                                              rois[ind_roi].voxel_index[:,1],
                                              rois[ind_roi].voxel_index[:,2]], axis=None)

            elif type == "SVM":
                roi_vect = vol.volume(vol.volume[rois[ind_roi].voxel_index[:,0],
                                                 rois[ind_roi].voxel_index[:,1],
                                                 rois[ind_roi].voxel_index[:,2]])
                weight_vect = rois[ind_roi].weights[rois[ind_roi].voxel_index[:,0],
                                                    rois[ind_roi].voxel_index[:,1],
                                                    rois[ind_roi].voxel_index[:,2]]
                ts_value = np.dot(roi_vect,weight_vect)

            if self.raw_time_series[ind_roi] is None:
                self.raw_time_series[ind_roi] = np.array([ts_value])
            else:
                self.raw_time_series[ind_roi] = np.append(self.raw_time_series[ind_roi], ts_value)

    def motion_correction_parameters(self, reference_mat, vol_mat):

        mot_corr_matrix = np.linalg.solve(reference_mat.T, vol_mat.T).T
        mot_corr_params = np.array(spm_imatrix(mot_corr_matrix), ndmin=2).T

        if len(self.offset_mc) == 0:
            self.offset_mc = mot_corr_params[0:6]

        if len(self.mc_params) == 0:
            self.mc_params = mot_corr_params[0:6] - self.offset_mc
        else:
            self.mc_params = np.hstack((self.mc_params, mot_corr_params[0:6] - self.offset_mc))

    def preprocessing(self, ind_vol_norm):

        for i_roi in range(self.nr_rois):

            # 1. Limits for scaling
            self.init_lim[i_roi] = 0.005 * np.mean(self.raw_time_series[i_roi])

            # Raw for display
            if self.disp_raw_time_series[i_roi] is None:
                self.disp_raw_time_series[i_roi] = np.array([0])
            else:
                self.disp_raw_time_series[i_roi] = np.append(self.disp_raw_time_series[i_roi],
                                                             self.raw_time_series[i_roi][-1]-self.raw_time_series[i_roi][0])

            # 2. Cumulative GLM
            tmp_ind_end = ind_vol_norm
            tmp_begin = 0
            # !!!
            # tmp_raw_time_series = self.raw_time_series[i_roi][tmp_begin:tmp_ind_end]

            # 6 MC regressors, linear trend, constant
            nr_regr_to_correct = 8

            # 2.1 time-series AR(1) filtering
            # if con.cglm_ar1:
            #
            #     if tmp_ind_end == 0:
            #         self.raw_time_series_ar1[i_roi] = (1- con.a_ar1) * tmp_raw_time_series[tmp_ind_end]
            #     else:
            #         self.raw_time_series_ar1[i_roi] = np.append(self.raw_time_series_ar1[i_roi],
            #                                                     tmp_raw_time_series[tmp_ind_end] -
            #                                                     con.a_ar1 * self.raw_time_series_ar1[i_roi][tmp_ind_end-1])
            #
            #     tmp_raw_time_series = self.raw_time_series_ar1[i_roi][tmp_begin:tmp_ind_end]
            #
            # # 2.2. exemplary step-wise addition of regressors, step = total nr of
            # # Regressors, which may require a justification for particular project
            # regr_step = nr_regr_to_correct


