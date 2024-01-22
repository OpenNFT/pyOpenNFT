import nibabel
from pathlib import PurePath
import numpy as np
import scipy.optimize as opt

from loguru import logger

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
class MrROI():
    # Префикс модальности позволит нам потом добавить eeg, nirs и тд
    """Contains single ROI
    """

    # --------------------------------------------------------------------------
    def __init__(self):
        self.volume = []
        self.mat = []
        self.dim = []
        self.voxel_index = []
        self.name = []
        self.weights = []

    # --------------------------------------------------------------------------
    def load_roi(self, file_path):
        image = nibabel.load(file_path, mmap=False)
        self.volume = np.array(image.get_fdata(), order='F')
        self.dim = np.array(image.shape)

        mat = image.affine
        # mat: a 12-parameter affine transform (from sform0)
        #      Note that the mapping is from voxels (where the first
        #      is considered to be at [1,1,1], to millimetres
        self.mat = mat @ np.hstack((np.eye(4, 3), np.array([-1, -1, -1, 1], ndmin=2).T))

        self.volume[np.nonzero(self.volume < 0.5)] = 0
        self.volume[np.nonzero(self.volume >= 0.5)] = 1

        self.voxel_index = np.argwhere(self.volume != 0)

        self.name = PurePath(file_path).parts[-1].split('.')[0]

    # --------------------------------------------------------------------------
    def load_weights(self, weigths_path):
        weight_image = nibabel.load(weigths_path, mmap=False)
        self.weights = np.array(weight_image.get_fdata(), order='F')

    # --------------------------------------------------------------------------
    def load_whole_brain_mask(self, epi_template):
        self.name = "Whole brain"
        self.mat = epi_template.mat
        self.dim = epi_template.dim

        epi_template.smooth()

        # mask voxels
        wb_mask_threshold = 30
        smoothed_volume = epi_template.volume
        smoothed_volume[np.nonzero(smoothed_volume < wb_mask_threshold)] = 0

        # histogram
        next_pow2 = np.ceil(np.log2(np.max(smoothed_volume[:, :, :])))
        nbins = np.int32(2 ** next_pow2)
        n, bns = np.histogram(smoothed_volume, bins=nbins)
        edges = np.arange(1, nbins+1)

        # histogram fit with single exponent and single gaussian
        fun = lambda a, b, c, d, e, x: a * np.exp(-x * b) + c * np.exp((-(x - d) ** 2) / e)
        # data
        xdata = edges[wb_mask_threshold-1:-1]
        lxdata = len(xdata)
        ydata = n[wb_mask_threshold-1:-1]

        # fit
        lb = np.array([1, 0, 0, 5, nbins / 10, nbins / 100])
        ub = np.array([np.inf, np.inf, np.inf, lxdata, np.inf])
        # maxfuneval = maxiter*(length(cf)+1);
        # opt = optimset('Tolfun',tolfun,'MaxFunEval',maxfuneval,'MaxIter',...
        #     maxiter,'Display','iter','DiffMinChange',1e-8);
        #
        # fitResults = lsqcurvefit(fun,cf,xdata,ydata,lb,ub,opt);
        fit_results = opt.curve_fit(fun, xdata, ydata, p0=[ydata[0], 1e-3, 10, np.round(lxdata / 2), 10],
                                    method='lm', ftol=1e-10)

        thr_wb_mask = np.round(fit_results[0][3]/2)

        # get mask
        wb_mask = np.zeros(self.dim)
        indexes_wb_mask = np.argwhere(smoothed_volume > thr_wb_mask)
        wb_mask[indexes_wb_mask[:, 0], indexes_wb_mask[:, 1], indexes_wb_mask[:, 2]] = 1

        # assign ROI
        self.volume = wb_mask
        self.voxel_index = indexes_wb_mask

        # DVARS scaling is most frequent image value given fit
        return np.median(smoothed_volume[indexes_wb_mask[:, 0], indexes_wb_mask[:, 1], indexes_wb_mask[:, 2]])
