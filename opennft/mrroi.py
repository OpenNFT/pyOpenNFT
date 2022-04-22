import nibabel
from pathlib import PurePath
import numpy as np

from loguru import logger


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

    def load_weights(self, weigths_path):
        weight_image = nibabel.load(weigths_path, mmap=False)
        self.weights = np.array(weight_image.get_fdata(), order='F')




