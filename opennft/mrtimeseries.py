import numpy as np

from rtspm import spm_imatrix
from loguru import logger


# --------------------------------------------------------------------------
class MrTimeSeries():

    # --------------------------------------------------------------------------
    def __init__(self, nr_rois):

        self.raw_time_series = [None] * nr_rois
        self.proc_time_series = [None] * nr_rois
        self.glm_time_series = [None] * nr_rois
        self.no_reg_time_series = [None] * nr_rois
        self.norm_time_series = [None] * nr_rois

        # x, y, z, pitch, roll, yaw
        self.mc_params = []
        self.offset_mc = []

    def acquiring(self, type, vol, rois):

        nr_rois = len(rois)

        for ind_roi in range(nr_rois):

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
                self.raw_time_series[ind_roi] = ts_value
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
