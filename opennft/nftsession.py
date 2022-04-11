# -*- coding: utf-8 -*-
import numpy as np

from pathlib import Path
from loguru import logger
from scipy.io import savemat

from opennft.mrvol import MrVol
from opennft.mrroi import MrROI
from opennft.mrtimeseries import MrTimeSeries
from opennft.utils import get_mosaic_dim, img_2d_to_3d


# --------------------------------------------------------------------------
class NftSession():
    """Session contains data like P structure
    """

    # --------------------------------------------------------------------------
    def __init__(self, config):

        self.config = config
        self.reference_vol = MrVol()  # mc_template
        self.vect_end_cond = np.ones((self.config.volumes_nr - self.config.skip_vol_nr, 1))
        self.first_nf_inds = []
        self.nr_rois = 0
        self.rois = []
        self.xdim_img_count = 0  # number of image in mosaic per horizontal
        self.ydim_img_count = 0  # number of image in mosaic per vertical
        self.img2d_dimx = 0      # mosaic image size X
        self.img2d_dimy = 0      # mosaic image size Y

    def setup(self):

        mc_templ_path = self.config.mc_template_file
        self.reference_vol.load_vol(mc_templ_path, "nii")
        self.xdim_img_count, self.ydim_img_count,  self.img2d_dimx,  self.img2d_dimy = get_mosaic_dim(self.reference_vol.dim)
        self.select_rois()

    def set_protocol(self, simulation_protocol):

        conditions = simulation_protocol["ConditionIndex"]
        cond_length = len(conditions)

        prot_names = []
        inc = 2
        for i in range(cond_length):
            prot_names.append(conditions[i]["ConditionName"])

            # TODO: check if baseline field already exists in protocol
            offsets = conditions[i]["OnOffsets"]

            for j in range(len(offsets)):
                self.vect_end_cond[offsets[j][0]-1:offsets[j][1]] = inc

                # for NFB blocks
                if inc == 2:
                    self.first_nf_inds.append(offsets[j][0]-1)

            inc = inc + 1



        # TODO: contrast


    def select_rois(self):

        for roi_file in Path(self.config.roi_files_dir).iterdir():
            if roi_file.is_file():
                self.nr_rois += 1
                new_roi = MrROI()
                new_roi.load_roi(roi_file.absolute())
                self.rois.append(new_roi)

        if self.config.type == "SVM":
            for weight_file, ind_roi in zip(Path(self.config.weights_file_name).iterdir(), range(self.nr_rois)):
                self.rois[ind_roi].load_weights(weight_file)


# --------------------------------------------------------------------------
class NftIteration():
    """Iteration contains data like main_loop_data
    """

    # --------------------------------------------------------------------------
    def __init__(self, session):
        self.pre_iter = -1
        self.iter_number = 0
        self.iter_norm_number = 0
        self.mr_vol = MrVol()
        self.mr_time_series = MrTimeSeries(session.nr_rois)
        self.session = session

        # realigment parameters
        self.a0 = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.deg = []
        self.b = []

    # --------------------------------------------------------------------------
    def load_vol(self, file_name, im_type):
        # возможно стоит сделать по задел под мультимодальность
        self.mr_vol.load_vol(file_name, im_type)
        self.mr_vol.volume = np.array(img_2d_to_3d(self.mr_vol.volume, self.session.xdim_img_count,
                                          self.session.ydim_img_count, self.session.reference_vol.dim), order='F')

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

    # --------------------------------------------------------------------------
    # test function
    def save_time_series(self):

        if len(self.mr_time_series.mc_params) == 0:
            logger.info(f"Empty data, nothing to save")
            return

        savemat("py_time_series.mat", {"raw_time_series": self.mr_time_series.raw_time_series[0],
                                       "x": self.mr_time_series.mc_params[0,:],
                                       "y": self.mr_time_series.mc_params[1,:],
                                       "z": self.mr_time_series.mc_params[2,:],
                                       "pitch": self.mr_time_series.mc_params[3,:],
                                       "roll": self.mr_time_series.mc_params[4,:],
                                       "yaw": self.mr_time_series.mc_params[5,:],
                                       })
