import nibabel
import pydicom
import numpy as np

from loguru import logger

from rtspm import spm_realign_rt, spm_reslice_rt, spm_smooth


# --------------------------------------------------------------------------
class MrVol():
    # Префикс модальности позволит нам потом добавить eeg, nirs и тд
    """Contains volume
    """

    # --------------------------------------------------------------------------
    def __init__(self):

        self.volume = []
        self.mat = []
        self.dim = []
        self.type = []

    # --------------------------------------------------------------------------
    def load_vol(self, file_path, im_type):

        self.type = im_type
        if im_type == "nii":
            image = nibabel.load(file_path, mmap=False)
            self.volume = np.array(image.get_fdata(), order='F')
            self.dim = np.array(image.shape)

            mat = image.affine
            # mat: a 12-parameter affine transform (from sform0)
            #      Note that the mapping is from voxels (where the first
            #      is considered to be at [1,1,1], to millimetres
            mat = mat @ np.hstack((np.eye(4, 3), np.array([-1, -1, -1, 1], ndmin=2).T))

            self.mat = mat

        elif im_type == "dcm":
            self.volume = np.array(pydicom.dcmread(file_path).pixel_array, order='F')

    # --------------------------------------------------------------------------
    def realign(self, iteration, a0, x1, x2, x3, deg, b):

        r = [{'mat': np.array([]), 'dim': np.array([]), 'Vol': np.array([])} for _ in range(2)]

        # на будущее
        # хотя добавление zero padding скорее подходит
        # в отдельный метод при чтении/инициализации/сетапе
        # is_zero_padding = iteration.session.config.zero_padding
        # nr_zero_pad_vol = iteration.session.config.nr_zero_padding

        # а может структуру R сформировать как переменную на этапе чтения вольюма?
        # со всеми сопутствующими padding'ами
        r[0]["Vol"] = iteration.session.reference_vol.volume
        r[0]["mat"] = iteration.session.reference_vol.mat
        r[0]["dim"] = iteration.session.reference_vol.dim

        r[1]["Vol"] = self.volume
        r[1]["mat"] = r[0]["mat"]
        r[1]["dim"] = r[0]["dim"]

        ind_vol = iteration.iter_number

        flags_spm_realign = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4,
                                  'wrap': np.zeros((3, 1)), 'rtm': 0, 'PW': '', 'lkp': np.array(range(0, 6))})

        [r, a0, x1, x2, x3, deg, b, _] = spm_realign_rt(r, flags_spm_realign, ind_vol + 1, 1, a0, x1, x2, x3, deg, b)

        # перезаписывать или хранить?
        self.volume = r[1]["Vol"]
        self.mat = r[1]["mat"]
        self.dim = r[1]["dim"]

        return r, a0, x1, x2, x3, deg, b

    def reslice(self, r):

        flags_spm_reslice = dict({'quality': .9, 'fwhm': 5, 'sep': 4, 'interp': 4,
                                  'wrap': np.zeros((3, 1)), 'mask': 1, 'mean': 0, 'which': 2})

        # перезаписывать или хранить?
        self.volume = spm_reslice_rt(r, flags_spm_reslice)

    def smooth(self):

        mat = self.mat
        dicom_info_vox = (np.sum(mat[0:3, 0:3] ** 2, axis=0)) ** .5

        gkernel = np.array([5, 5, 5]) / dicom_info_vox

        # перезаписывать или хранить?
        self.volume = spm_smooth(self.volume, gkernel)



# --------------------------------------------------------------------------
class MrROI():
    """Contains single ROI
    """
    pass


# --------------------------------------------------------------------------
class MrReferenceVol():
    """Contains registration reference volume for motion correction
        aka mc_template
    """
    pass
