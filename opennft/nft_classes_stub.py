# -*- coding: utf-8 -*-
"""
Общие мысли:
1.  Все собрал в одном файле только для лучшей читаемости на уровне прототипа.
    Потом надо разделить.
2.  Надо подумать как это назвать, использовать ли префикс nft
3.
"""

from pathlib import Path
from loguru import logger


# Классы "процессы"

# --------------------------------------------------------------------------
class Config():
    """All config-related magick is here
    """
    pass


# --------------------------------------------------------------------------
class NftSession():
    """Session contains data like P structure
    """

    # --------------------------------------------------------------------------
    def __init__(self, config: Config):
        reference_vol = None # mc_template
        pass


# --------------------------------------------------------------------------
class NftIteration():
    """Iteration contains data like main_loop_data
    """

    # --------------------------------------------------------------------------
    def __init__(self, session):

        self.iter_number = 0
        self.mr_vol = MrVol()
        self.session = session

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
    def load_vol(self, file_name):
        # возможно стоит сделать по задел под мультимодальность
        self.mr_vol.load_vol(file_name)

    # --------------------------------------------------------------------------
    def process_vol(self):
        self.mr_vol.coregister(self)
        #self.mr_vol.interpolate(self)
        #self.mr_vol.smooth(self)


# Классы объекты
# --------------------------------------------------------------------------
class MrVol():
    # Префикс модальности позмолит нам потом добавить eeg, nirs и тд
    """Contains volume
    """

    # --------------------------------------------------------------------------
    def __init__(self):
        pass

    # --------------------------------------------------------------------------
    def load_vol(self, file_name):
        logger.info(f"Load vol {file_name}")
        pass

    # --------------------------------------------------------------------------
    def coregister(self, iteration):
        # get reference vol from iteration.session.reference_vol
        logger.info(f"coregister to reference vol")

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


# --------------------------------------------------------------------------
class MrSignal():
    """Contains signal (time series) extracted from Vol
    """
    pass


class Drawer():
    """Draws data with GUI
    """
    pass


# --------------------------------------------------------------------------
class MrFeedback():
    """All feedback magic is here. Or not?
    """
    pass
