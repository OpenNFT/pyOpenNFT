import enum
import time
import sys
import numpy as np
import pyqtgraph as pg
import multiprocessing as mp

from multiprocessing import shared_memory
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget
from loguru import logger

import opennft_calc
from opennft import mosaicview, projview, mapimagewidget, volviewformation
from opennft.config import config as con


class ImageViewMode(enum.IntEnum):
    mosaic = 0
    orthview_anat = 1
    orthview_epi = 2


class OpenNFTCore(QWidget):
    def __init__(self, *args, **kwargs):

        self.init_service_data()

        self._calc_process = opennft_calc.OpenNFTCalc(self._service_data)

        self.nr_vol = self._service_data["nr_vol"]
        self.nr_rois = self._service_data["nr_rois"]
        self.vol_dim = self._service_data["vol_dim"]
        self.mosaic_dim = self._service_data["mosaic_dim"]
        self.view_form_init()

        self.init_shm()

        if con.use_gui:
            super().__init__(*args, **kwargs)

            loadUi('opennft.ui', self)

            self.mosaicImageView = mosaicview.MosaicImageViewWidget(self)
            self.layoutMosaic.addWidget(self.mosaicImageView)

            self.orthView = projview.ProjectionsWidget(self)
            self.layoutOrthView.addWidget(self.orthView)

            self.pos_map_thresholds_widget = mapimagewidget.MapImageThresholdsWidget(self)
            self.neg_map_thresholds_widget = mapimagewidget.MapImageThresholdsWidget(self, colormap='Blues_r')

            self.layoutHotMapThresholds.addWidget(self.pos_map_thresholds_widget)
            self.pos_map_thresholds_widget.setEnabled(False)
            self.layoutNegMapThresholds.addWidget(self.neg_map_thresholds_widget)
            self.neg_map_thresholds_widget.setEnabled(False)

            self.cbImageViewMode.currentIndexChanged.connect(self.onChangeImageViewMode)
            self.cbImageViewMode.setEnabled(True)
            self.orthView.cursorPositionChanged.connect(self.onChangeOrthViewCursorPosition)

            self.mcPlot = self.createMcPlot(self.layoutPlot1)
            self.btnStart.clicked.connect(self.onStart)
            self.guiTimer = QTimer(self)
            self.guiTimer.timeout.connect(self.onCheckGUIUpdated)
            self.show()
        else:
            self.onStart()

    def init_service_data(self):

        self._service_data = mp.Manager().dict()

        self._service_data["data_ready_flag"] = False
        self._service_data["init"] = False
        self._service_data["nr_vol"] = 0
        self._service_data["nr_rois"] = 0
        self._service_data["vol_dim"] = 0
        self._service_data["mosaic_dim"] = 0
        self._service_data["is_ROI"] = con.use_roi
        self._service_data["vol_mat"] = None
        self._service_data["is_stopped"] = False
        self._service_data["ready_to_form"] = False
        self._service_data["view_mode"] = 0
        self._service_data["done_mosaic_templ"] = False
        self._service_data["done_orth"] = False
        self._service_data["overlay_ready"] = False

        self._service_data["proj_dims"] = None
        self._service_data["is_neg"] = False
        self._service_data["bg_type"] = "bgEPI"
        self._service_data["cursor_pos"] = (129, 95)
        self._service_data["flags_planes"] = projview.ProjectionType.coronal.value

    def init_shm(self):

        mc_array = np.zeros((self.nr_vol, 6), dtype=np.float32)
        self.mc_shm = shared_memory.SharedMemory(create=True, size=mc_array.nbytes, name=con.shm_file_names[0])
        self.mc_data = np.ndarray(shape=mc_array.shape, dtype=mc_array.dtype, buffer=self.mc_shm.buf)

        mosaic_array = np.zeros(self.mosaic_dim, dtype=np.float32)
        self.mosaic_shm = shared_memory.SharedMemory(create=True, size=mosaic_array.nbytes * 3, name=con.shm_file_names[1])
        self.mosaic_data = np.ndarray(shape=mosaic_array.shape, dtype=mosaic_array.dtype, buffer=self.mosaic_shm.buf)

        vol_array = np.zeros(self._service_data["vol_dim"], dtype=np.float32)
        self.epi_shm = shared_memory.SharedMemory(create=True, size=vol_array.nbytes, name=con.shm_file_names[2])
        self.epi_data = np.ndarray(shape=vol_array.shape, dtype=vol_array.dtype, buffer=self.epi_shm.buf)
        self.epi_data[:,:,:] = self._calc_process.session.reference_vol.volume

        overlay_array = np.zeros(self._service_data["vol_dim"], dtype=np.float32)
        self.overlay_shm = shared_memory.SharedMemory(create=True, size=overlay_array.nbytes, name=con.shm_file_names[3])
        self.overlay_data = np.ndarray(shape=overlay_array.shape, dtype=overlay_array.dtype, buffer=self.overlay_shm.buf)

        dims = self._service_data["proj_dims"]
        proj_array = np.zeros((dims[0],dims[1]), dtype=np.float32)
        self.proj_t_shm = shared_memory.SharedMemory(create=True, size=proj_array.nbytes, name=con.shm_file_names[5])
        self.proj_t = np.ndarray(shape=(dims[0],dims[1]),
                                 dtype=np.float32,
                                 buffer=self.proj_t_shm.buf,
                                 order="F")

        proj_array = np.zeros((dims[0],dims[2]), dtype=np.float32)
        self.proj_c_shm = shared_memory.SharedMemory(create=True, size=proj_array.nbytes, name=con.shm_file_names[6])
        self.proj_c = np.ndarray(shape=(dims[0],dims[2]),
                                 dtype=np.float32,
                                 buffer=self.proj_c_shm.buf,
                                 order="F")

        proj_array = np.zeros((dims[1],dims[2]), dtype=np.float32)
        self.proj_s_shm = shared_memory.SharedMemory(create=True, size=proj_array.nbytes, name=con.shm_file_names[7])
        self.proj_s = np.ndarray(shape=(dims[1],dims[2]),
                                 dtype=np.float32,
                                 buffer=self.proj_s_shm.buf,
                                 order="F")

    def view_form_init(self):

        self._orth_view = volviewformation.VolViewFormation(self._service_data)
        self._orth_view.start()

    def createMcPlot(self, layoutPlot):
        mctrotplot = pg.PlotWidget(self)
        mctrotplot.setBackground((255, 255, 255))
        layoutPlot.addWidget(mctrotplot)

        p = mctrotplot.getPlotItem()
        p.setTitle('MC', size='')
        p.setLabel('left', "Amplitude [a.u.]")
        p.setMenuEnabled(enableMenu=False)
        p.setMouseEnabled(x=False, y=False)
        p.installEventFilter(self)

        p.disableAutoRange(axis=pg.ViewBox.XAxis)
        p.setXRange(1, 150, padding=0.0)
        p.showGrid(x=True, y=True, alpha=0.7)

        return mctrotplot

    def drawMcPlots(self, mcPlot, data):

        mctrrot = mcPlot.getPlotItem()

        # if self._service_data["init"]:
        mctrrot.clear()

        plots = []

        MC_PLOT_COLORS = [
            (255, 123, 0),  # translations - x, y, z
            (255, 56, 109),
            (127, 0, 255),
            (0, 46, 255),  # rotations - alpha, betta, gamma
            (0, 147, 54),
            (145, 130, 43),
        ]

        for i, c in enumerate(MC_PLOT_COLORS):
            plots.append(mctrrot.plot(pen=c))

        self.drawMcPlots.__dict__['mctrrot'] = plots

        x = np.arange(1, data.shape[0] + 1, dtype=np.float64)

        for pt, i1, in zip(
                self.drawMcPlots.__dict__['mctrrot'], range(0, 6)):
            pt.setData(x=x, y=data[:, i1])

    def onStart(self):
        if not self._calc_process.is_alive():
            print("main starting process")
            self._calc_process.start()
            if con.use_gui:
                self.guiTimer.start(30)
        else:
            pass

    # --------------------------------------------------------------------------
    def onChangeImageViewMode(self, index):
        if index == 0:
            stack_index = 0
            mode = ImageViewMode.mosaic
        elif index == 1:
            stack_index = 1
            mode = ImageViewMode.orthview_anat
        else:
            stack_index = 1
            mode = ImageViewMode.orthview_epi

        self.stackedWidgetImages.setCurrentIndex(stack_index)
        self._service_data["view_mode"] = mode

        # if self.cbImageViewMode.isEnabled():
        #     self.updateOrthViewAsync()
        #     self.onInteractWithMapImage()

    # --------------------------------------------------------------------------
    def onChangeOrthViewCursorPosition(self, pos, proj):
        self._service_data["cursor_pos"] = pos
        self._service_data["flags_planes"] = proj.value

        logger.debug('New cursor coords {} for proj "{}" have been received', pos, proj.name)

    def onCheckGUIUpdated(self):

        if self._service_data["data_ready_flag"]:
            self.drawMcPlots(self.mcPlot, self.mc_data)
            self._service_data["data_ready_flag"] = False

        if self._service_data["view_mode"] == ImageViewMode.mosaic :

            if self._service_data["done_mosaic_templ"]:

                self.onCheckMosaicViewUpdated()

        else:

            if self._service_data["done_orth"]:

                self.onCheckOrthViewUpdated()
                self._service_data["done_orth"] = False

    def onCheckMosaicViewUpdated(self):

        if self._service_data["done_mosaic_templ"]:
            background_image = np.ndarray(shape=self.mosaic_dim, dtype=np.float32, buffer=self.mosaic_shm.buf)
            if background_image.size > 0:
                logger.info("Done mosaic template")
                self.mosaicImageView.set_background_image(background_image)
            else:
                return

            self._service_data["done_mosaic_templ"] = False

    def onCheckOrthViewUpdated(self):

        for proj in projview.ProjectionType:

            if proj == projview.ProjectionType.transversal:
                bg_image = self.proj_t.T
            elif proj == projview.ProjectionType.sagittal:
                bg_image = self.proj_s.T
            elif proj == projview.ProjectionType.coronal:
                bg_image = self.proj_c.T

            self.orthView.set_background_image(proj, bg_image)

    def closeEvent(self, event):
        if self._calc_process.is_alive():
            self._calc_process.join()
        if self._orth_view.is_alive():
            self._service_data["is_stopped"] = True
            self._orth_view.join()

        self.mc_shm.close()
        self.mc_shm.unlink()
        self.mosaic_shm.close()
        self.mosaic_shm.unlink()
        self.epi_shm.close()
        self.epi_shm.unlink()
        self.stat_shm.close()
        self.stat_shm.unlink()
        self.proj_t_shm.close()
        self.proj_t_shm.unlink()
        self.proj_c_shm.close()
        self.proj_c_shm.unlink()
        self.proj_s_shm.close()
        self.proj_s_shm.unlink()

        self._service_data = None
        self.close()
        print("main process finished")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = OpenNFTCore()
    sys.exit(app.exec_())
