import numpy as np

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
        self.mc_params = [[], [], [], [], [], []]

    def acquiring(self, type, vol, rois):

        nr_rois = len(rois)

        for ind_roi in range(nr_rois):

            if type == "PSC":
                ts_value = np.mean(vol.volume[rois[ind_roi].voxel_index], axis=None)

            elif type == "SVM":
                roi_vect = vol.volume(rois[ind_roi].voxel_index)
                weight_vect = rois[ind_roi].weights(rois[ind_roi].voxel_index)
                ts_value = np.dot(roi_vect,weight_vect)

            self.raw_time_series[ind_roi] = np.append(self.raw_time_series[ind_roi], ts_value)
