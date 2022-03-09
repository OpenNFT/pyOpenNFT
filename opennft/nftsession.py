# -*- coding: utf-8 -*-

from pathlib import Path
from loguru import logger
from scipy.io import savemat

from opennft.mrvol import MrVol
from opennft.mrroi import MrROI
from opennft.mrtimeseries import MrTimeSeries
import opennft.nft_classes_stub as nft


# --------------------------------------------------------------------------
class NftSession():
    """Session contains data like P structure
    """

    # --------------------------------------------------------------------------
    def __init__(self, config):

        self.config = config
        self.reference_vol = MrVol()  # mc_template
        self.nr_rois = 0
        self.rois = []

    def setup(self):

        mc_templ_path = self.config.mc_template_file
        self.reference_vol.load_vol(mc_templ_path, "nii")
        self.select_rois()

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
        self.iter_number = 0
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

        # вот такая у меня мысль с набором хендлеров
        # возможные косяки -
        # 0. возможно не стоит так заморачиваться и подвесить плагины на RTTI
        #    сделать такой массив хендлеров только для плагинов, а когда их вызывать
        #    определять по декоратору - типа https://github.com/gahjelle/pyplugs
        # 1. сделать insert сложно, но возможно все плагины надо вставлять на этапе создания
        # 2. весь функционал хендлеров стоит сделать в базовом классе, и от него наследовать Сессию и Итерацию
        self.handlers = {}
        self.handlers['load_scan'] = {}
        self.handlers['process_scan'] = {}
        self.handlers['compute_signals'] = {}
        self.handlers['compute_feedback'] = {}

        self.handlers_data = {}
        self.handlers_data['load_scan'] = {}
        self.handlers_data['process_scan'] = {}
        self.handlers_data['compute_signals'] = {}
        self.handlers_data['compute_feedback'] = {}

        self.set_basic_handlers()

    # --------------------------------------------------------------------------
    def set_basic_handlers(self):
        self.handlers['load_scan']['load_mr_vol'] = MrVol.load_vol

        self.handlers_data['load_scan']['load_mr_vol'] = None

    # --------------------------------------------------------------------------
    def dispatch_handlers(self):
        self.iter_number += 1

        # понятное дело что надо сделать цикл по всем хендлерам
        self.handlers['load_scan']['load_mr_vol'](*self.handlers_data['load_scan']['load_mr_vol'])

    # или без выпендрежа с хендлерами
    # --------------------------------------------------------------------------
    def load_vol(self, file_name, im_type):
        # возможно стоит сделать по задел под мультимодальность
        self.mr_vol.load_vol(file_name, im_type)

    # --------------------------------------------------------------------------
    def process_vol(self):
        r, self.a0, self.x1, self.x2, self.x3, self.deg, self.b = self.mr_vol.realign(self, self.a0, self.x1, self.x2,
                                                                                      self.x3, self.deg, self.b)
        self.mr_time_series.motion_correction_parameters(r[0]["mat"], r[1]["mat"])
        self.mr_vol.reslice(r)
        self.mr_vol.smooth()

    def process_time_series(self):
        self.mr_time_series.acquiring(self.session.config.type, self.mr_vol, self.session.rois)

    def save_time_series(self):
        savemat("py_time_series.mat", {"raw_time_series": self.mr_time_series.raw_time_series[0],
                                       "x": self.mr_time_series.mc_params[0,:],
                                       "y": self.mr_time_series.mc_params[1,:],
                                       "z": self.mr_time_series.mc_params[2,:],
                                       "pitch": self.mr_time_series.mc_params[3,:],
                                       "roll": self.mr_time_series.mc_params[4,:],
                                       "yaw": self.mr_time_series.mc_params[5,:],
                                       })
