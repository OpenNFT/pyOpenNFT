import numpy as np
from opennft.config import config as con
from opennft.utils import ar_regr, zscore
from opennft.modif_kalman import modif_kalman
from opennft.scale_time_series import scale_time_series

from rtspm import spm_imatrix


# --------------------------------------------------------------------------
class MrTimeSeries():

    # --------------------------------------------------------------------------
    def __init__(self, nr_rois):

        self.raw_time_series = [None] * nr_rois
        self.disp_raw_time_series = [None] * nr_rois
        self.raw_time_series_ar1 = [None] * nr_rois
        self.init_lim = [None] * nr_rois
        self.kalman_proc_time_series = [None] * nr_rois
        self.glm_time_series = [None] * nr_rois
        self.no_reg_time_series = [None] * nr_rois
        self.scale_time_series = [None] * nr_rois
        self.lin_trend_betas = [None] * nr_rois
        self.nr_rois = nr_rois

        # x, y, z, pitch, roll, yaw
        self.mc_params = []
        self.offset_mc = []

        # Kalman presets
        self.s = {"Q": np.zeros((nr_rois, 1)), "P": np.zeros((nr_rois, 1)),
                  "R": np.zeros((nr_rois, 1)), "x": np.zeros((nr_rois, 1))}
        self.flag_pos_deriv_spike = np.zeros((nr_rois, 1))
        self.flag_neg_deriv_spike = np.zeros((nr_rois, 1))

        # Scaling
        self.tmp_pos_min = np.zeros((nr_rois, 1))
        self.tmp_pos_max = np.zeros((nr_rois, 1))
        self.pos_min = [None] * nr_rois
        self.pos_max = [None] * nr_rois
        self.output_pos_min = []
        self.output_pos_max = []

    def acquiring(self, type, vol, rois):

        for ind_roi in range(self.nr_rois):

            if type == "PSC":
                ts_value = np.mean(vol.volume[rois[ind_roi].voxel_index[:, 0],
                rois[ind_roi].voxel_index[:, 1],
                rois[ind_roi].voxel_index[:, 2]], axis=None)

            elif type == "SVM":
                roi_vect = vol.volume[rois[ind_roi].voxel_index[:, 0],
                rois[ind_roi].voxel_index[:, 1],
                rois[ind_roi].voxel_index[:, 2]]
                weight_vect = rois[ind_roi].weights[rois[ind_roi].voxel_index[:, 0],
                rois[ind_roi].voxel_index[:, 1],
                rois[ind_roi].voxel_index[:, 2]]
                ts_value = np.dot(roi_vect, weight_vect)

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

    def preprocessing(self, ind_vol_norm, bas_func, lin_regr, sl_wind, vect_end_cond, bas_block_length, is_svm):

        nr_bas_func = len(bas_func[0])

        for i_roi in range(self.nr_rois):

            # 1. Limits for scaling
            self.init_lim[i_roi] = 0.005 * np.mean(self.raw_time_series[i_roi])

            # Raw for display
            if self.disp_raw_time_series[i_roi] is None:
                self.disp_raw_time_series[i_roi] = np.array([0])
            else:
                self.disp_raw_time_series[i_roi] = np.append(self.disp_raw_time_series[i_roi],
                                                             self.raw_time_series[i_roi][-1] -
                                                             self.raw_time_series[i_roi][0])

            # 2. Cumulative GLM
            tmp_ind_end = ind_vol_norm
            tmp_begin = 0
            # !!!
            tmp_raw_time_series = self.raw_time_series[i_roi][tmp_begin:tmp_ind_end + 1]

            # 6 MC regressors, linear trend, constant
            nr_regr_to_correct = 8

            # 2.1 time-series AR(1) filtering
            if con.cglm_ar1:

                if tmp_ind_end == 0:
                    self.raw_time_series_ar1[i_roi] = np.array((1 - con.a_ar1) * tmp_raw_time_series[tmp_ind_end],
                                                               ndmin=1)
                else:
                    self.raw_time_series_ar1[i_roi] = np.append(self.raw_time_series_ar1[i_roi],
                                                                tmp_raw_time_series[tmp_ind_end] -
                                                                con.a_ar1 * self.raw_time_series_ar1[i_roi][
                                                                    tmp_ind_end - 1])

                tmp_raw_time_series = self.raw_time_series_ar1[i_roi][tmp_begin:tmp_ind_end + 1]

            tmp_raw_time_series = np.array(tmp_raw_time_series, ndmin=2).transpose()

            if ind_vol_norm == 0:
                self.mc_params[:, ind_vol_norm] = 1e-5

            # # 2.2. exemplary step-wise addition of regressors, step = total nr of
            # # Regressors, which may require a justification for particular project
            regr_step = nr_bas_func + nr_regr_to_correct
            if tmp_ind_end < regr_step - 1:

                tmp_regr = np.ones((tmp_ind_end + 1, 1))
                if con.cglm_ar1:
                    tmp_regr = ar_regr(con.a_ar1, tmp_regr)
                cx0 = tmp_regr
                beta_reg = np.linalg.pinv(cx0) @ tmp_raw_time_series
                tmp_glm_proc_time_series = (tmp_raw_time_series - cx0 * beta_reg)

            elif (tmp_ind_end >= regr_step - 1) and (tmp_ind_end < 2 * regr_step - 1):

                tmp_regr = np.hstack((np.ones((tmp_ind_end + 1, 1)), lin_regr[0:tmp_ind_end + 1]))
                if con.cglm_ar1:
                    tmp_regr = ar_regr(con.a_ar1, tmp_regr)
                cx0 = tmp_regr
                beta_reg = np.linalg.pinv(cx0) @ tmp_raw_time_series
                tmp_glm_proc_time_series = (tmp_raw_time_series - cx0 @ beta_reg)

            elif (tmp_ind_end >= 2 * regr_step - 1) and (tmp_ind_end < 3 * regr_step - 1):

                tmp_regr = np.hstack((np.ones((tmp_ind_end + 1, 1)), lin_regr[0:tmp_ind_end + 1]))
                tmp_regr = np.hstack((tmp_regr, zscore(self.mc_params[:, 0:tmp_ind_end + 1].T)))
                if con.cglm_ar1:
                    tmp_regr = ar_regr(con.a_ar1, tmp_regr)
                cx0 = tmp_regr
                beta_reg = np.linalg.pinv(cx0) @ tmp_raw_time_series
                tmp_glm_proc_time_series = (tmp_raw_time_series - cx0 @ beta_reg)

            else:

                tmp_regr = np.hstack((np.ones((tmp_ind_end + 1, 1)), lin_regr[0:tmp_ind_end + 1]))
                tmp_regr = np.hstack((tmp_regr, zscore(self.mc_params[:, 0:tmp_ind_end + 1].T)))
                if con.cglm_ar1:
                    tmp_regr = ar_regr(con.a_ar1, tmp_regr)
                cx0 = np.hstack((tmp_regr, bas_func[0:tmp_ind_end + 1, :]))
                beta_reg = np.linalg.pinv(cx0) @ tmp_raw_time_series
                tmp_glm_proc_time_series = (tmp_raw_time_series - cx0 @
                                            np.vstack((beta_reg[0:-1 - nr_bas_func + 1], np.zeros((nr_bas_func, 1)))))
                if con.use_rtqa:
                    tmp_no_reg_glm_proc_time_series = (tmp_raw_time_series - cx0 @
                                                       np.vstack((np.zeros((len(beta_reg) - nr_bas_func, 1)),
                                                                  beta_reg[-1 - nr_bas_func + 1:])))

            if tmp_ind_end == 0:
                self.glm_time_series[i_roi] = np.array(tmp_glm_proc_time_series)
            else:
                self.glm_time_series[i_roi] = np.append(self.glm_time_series[i_roi], tmp_glm_proc_time_series[-1])

            if con.use_rtqa:

                if tmp_ind_end == 0:
                    self.no_reg_time_series[i_roi] = np.array(tmp_raw_time_series)
                elif tmp_ind_end < 3 * regr_step:
                    self.no_reg_time_series[i_roi] = np.append(self.no_reg_time_series[i_roi],
                                                               tmp_raw_time_series[-1])
                else:
                    self.no_reg_time_series[i_roi] = np.append(self.no_reg_time_series[i_roi],
                                                               tmp_no_reg_glm_proc_time_series[-1])
                if tmp_ind_end >= regr_step - 1:
                    self.lin_trend_betas[i_roi] = beta_reg[1]
                else:
                    self.lin_trend_betas[i_roi] = 0

            # 3. modified Kalman low-pass filter + spike identification & correction
            tmp_std = np.std(self.glm_time_series[i_roi])
            self.s["Q"][i_roi] = .25 * (tmp_std ** 2)
            self.s["R"][i_roi] = tmp_std ** 2
            kalman_threshold = .9 * tmp_std

            temp_s = {"Q": self.s["Q"][i_roi], "R": self.s["R"][i_roi], "P": self.s["P"][i_roi],
                      "x": self.s["x"][i_roi]}

            kalman_out, temp_s, self.flag_pos_deriv_spike[i_roi], self.flag_neg_deriv_spike[i_roi] = modif_kalman(
                kalman_threshold, self.glm_time_series[i_roi][ind_vol_norm],
                temp_s, self.flag_pos_deriv_spike[i_roi], self.flag_neg_deriv_spike[i_roi])

            if tmp_ind_end == 0:
                self.kalman_proc_time_series[i_roi] = np.array(kalman_out, ndmin=1)
            else:
                self.kalman_proc_time_series[i_roi] = np.append(self.kalman_proc_time_series[i_roi], kalman_out)

            self.s["Q"][i_roi] = temp_s["Q"]
            self.s["P"][i_roi] = temp_s["P"]
            self.s["x"][i_roi] = temp_s["x"]

            # 4. Scaling
            scale_out, self.tmp_pos_min[i_roi], self.tmp_pos_max[i_roi] = scale_time_series(
                self.kalman_proc_time_series[i_roi], ind_vol_norm, sl_wind, self.init_lim[i_roi],
                self.tmp_pos_min[i_roi], self.tmp_pos_max[i_roi], vect_end_cond, bas_block_length
            )

            if tmp_ind_end == 0:
                self.scale_time_series[i_roi] = np.array(scale_out, ndmin=1)
                self.pos_min[i_roi] = np.array(self.tmp_pos_min[i_roi], ndmin=1)
                self.pos_max[i_roi] = np.array(self.tmp_pos_max[i_roi], ndmin=1)
            else:
                self.scale_time_series[i_roi] = np.append(self.scale_time_series[i_roi], scale_out)
                self.pos_min[i_roi] = np.append(self.pos_min[i_roi], self.tmp_pos_min[i_roi])
                self.pos_max[i_roi] = np.append(self.pos_max[i_roi], self.tmp_pos_max[i_roi])

            # 5. z-scoring and sigmoidal transform
            if is_svm:
                zscored_val = zscore(self.scale_time_series[i_roi][0:ind_vol_norm + 1])

                self.scale_time_series[i_roi][ind_vol_norm] = 1 / (1 + np.exp(-zscored_val[-1]))

        mean_pos_min = np.mean(np.array(self.pos_min))
        mean_pos_max = np.mean(np.array(self.pos_max))

        if tmp_ind_end == 0:
            self.output_pos_min = self.pos_min.copy()
            self.output_pos_min.append(np.array(mean_pos_min, ndmin=1))
            self.output_pos_max = self.pos_max.copy()
            self.output_pos_max.append(np.array(mean_pos_max, ndmin=1))
        else:
            for i_roi in range(self.nr_rois):
                self.output_pos_min[i_roi] = np.append(self.output_pos_min[i_roi], self.pos_min[i_roi][-1])
                self.output_pos_max[i_roi] = np.append(self.output_pos_max[i_roi], self.pos_max[i_roi][-1])
            self.output_pos_min[-1] = np.append(self.output_pos_min[-1], mean_pos_min)
            self.output_pos_max[-1] = np.append(self.output_pos_max[-1], mean_pos_max)
