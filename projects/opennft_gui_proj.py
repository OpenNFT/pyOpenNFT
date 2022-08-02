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

import opennft_console_proj
from opennft import mosaicview, projview, mapimagewidget, volviewformation
from opennft.config import config as con


class ImageViewMode(str, enum.Enum):
    mosaic = 'mosaic'
    orthview_anat = 'orthview_anat'
    orthview_epi = 'orthview_epi'


class OpenNFTManager(QWidget):
    def __init__(self, *args, **kwargs):

        self.init_exchange_data()

        self._core_process = opennft_console_proj.OpenNFTCoreProj(self.exchange_data)

        self.nr_vol = self.exchange_data["nr_vol"]
        self.nr_rois = self.exchange_data["nr_rois"]
        self.vol_dim = self.exchange_data["vol_dim"]
        self.mosaic_dim = self.exchange_data["mosaic_dim"]
        self.view_form_init()

        self.init_shmem()

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

    def init_exchange_data(self):

        self.exchange_data = mp.Manager().dict()

        self.exchange_data["data_ready_flag"] = False
        self.exchange_data["init"] = False
        self.exchange_data["nr_vol"] = 0
        self.exchange_data["nr_rois"] = 0
        self.exchange_data["vol_dim"] = 0
        self.exchange_data["mosaic_dim"] = 0
        self.exchange_data["is_ROI"] = con.use_roi
        self.exchange_data["vol_mat"] = None
        self.exchange_data["is_stopped"] = False
        self.exchange_data["ready_to_form"] = False
        self.exchange_data["view_mode"] = 'mosaic'
        self.exchange_data["done_mosaic_templ"] = False
        self.exchange_data["done_orth"] = False
        self.exchange_data["overlay_ready"] = False

        self.exchange_data["proj_dims"] = None
        self.exchange_data["is_neg"] = False
        self.exchange_data["bg_type"] = "bgEPI"
        self.exchange_data["cursor_pos"] = (129, 95)
        self.exchange_data["flags_planes"] = projview.ProjectionType.coronal.value

    def init_shmem(self):

        mc_array = np.zeros((self.nr_vol, 6), dtype=np.float32)
        self.mc_shmem = shared_memory.SharedMemory(create=True, size=mc_array.nbytes, name=con.shmem_file_names[0])
        self.mc_data = np.ndarray(shape=mc_array.shape, dtype=mc_array.dtype, buffer=self.mc_shmem.buf)

        mosaic_array = np.zeros(self.mosaic_dim, dtype=np.float32)
        self.mosaic_shmem = shared_memory.SharedMemory(create=True, size=mosaic_array.nbytes * 3, name=con.shmem_file_names[1])
        self.mosaic_data = np.ndarray(shape=mosaic_array.shape, dtype=mosaic_array.dtype, buffer=self.mosaic_shmem.buf)

        vol_array = np.zeros(self.exchange_data["vol_dim"], dtype=np.float32)
        self.epi_shmem = shared_memory.SharedMemory(create=True, size=vol_array.nbytes, name=con.shmem_file_names[2])
        self.epi_data = np.ndarray(shape=vol_array.shape, dtype=vol_array.dtype, buffer=self.epi_shmem.buf)
        self.epi_data[:,:,:] = self._core_process.session.reference_vol.volume

        overlay_array = np.zeros(self.exchange_data["vol_dim"], dtype=np.float32)
        self.overlay_shmem = shared_memory.SharedMemory(create=True, size=overlay_array.nbytes, name=con.shmem_file_names[3])
        self.overlay_data = np.ndarray(shape=overlay_array.shape, dtype=overlay_array.dtype, buffer=self.overlay_shmem.buf)

        dims = self.exchange_data["proj_dims"]
        proj_array = np.zeros((dims[0],dims[1]), dtype=np.float32)
        self.proj_t_shmem = shared_memory.SharedMemory(create=True, size=proj_array.nbytes, name=con.shmem_file_names[5])
        self.proj_t = np.ndarray(shape=(dims[0],dims[1]),
                                 dtype=np.float32,
                                 buffer=self.proj_t_shmem.buf,
                                 order="F")

        proj_array = np.zeros((dims[0],dims[2]), dtype=np.float32)
        self.proj_c_shmem = shared_memory.SharedMemory(create=True, size=proj_array.nbytes, name=con.shmem_file_names[6])
        self.proj_c = np.ndarray(shape=(dims[0],dims[2]),
                                 dtype=np.float32,
                                 buffer=self.proj_c_shmem.buf,
                                 order="F")

        proj_array = np.zeros((dims[1],dims[2]), dtype=np.float32)
        self.proj_s_shmem = shared_memory.SharedMemory(create=True, size=proj_array.nbytes, name=con.shmem_file_names[7])
        self.proj_s = np.ndarray(shape=(dims[1],dims[2]),
                                 dtype=np.float32,
                                 buffer=self.proj_s_shmem.buf,
                                 order="F")

    def view_form_init(self):

        self._view_form_process = volviewformation.VolViewFormation(self.exchange_data)
        self._view_form_process.start()

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

        # if self.exchange_data["init"]:
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
        if not self._core_process.is_alive():
            print("main starting process")
            self._core_process.start()
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
        self.exchange_data["view_mode"] = mode

        # if self.cbImageViewMode.isEnabled():
        #     self.updateOrthViewAsync()
        #     self.onInteractWithMapImage()

    # --------------------------------------------------------------------------
    def onChangeOrthViewCursorPosition(self, pos, proj):
        self.exchange_data["cursor_pos"] = pos
        self.exchange_data["flags_planes"] = proj.value

        logger.debug('New cursor coords {} for proj "{}" have been received', pos, proj.name)

    def onCheckGUIUpdated(self):

        if self.exchange_data["data_ready_flag"]:
            self.drawMcPlots(self.mcPlot, self.mc_data)
            self.exchange_data["data_ready_flag"] = False

        if self.exchange_data["view_mode"] == ImageViewMode.mosaic :

            if self.exchange_data["done_mosaic_templ"]:

                self.onCheckMosaicViewUpdated()

        else:

            if self.exchange_data["done_orth"]:

                self.onCheckOrthViewUpdated()
                self.exchange_data["done_orth"] = False

    def onCheckMosaicViewUpdated(self):

        if self.exchange_data["done_mosaic_templ"]:
            background_image = np.ndarray(shape=self.mosaic_dim, dtype=np.float32, buffer=self.mosaic_shmem.buf)
            if background_image.size > 0:
                logger.info("Done mosaic template")
                self.mosaicImageView.set_background_image(background_image)
            else:
                return

            self.exchange_data["done_mosaic_templ"] = False

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
        if self._core_process.is_alive():
            self._core_process.join()
        if self._view_form_process.is_alive():
            self.exchange_data["is_stopped"] = True
            self._view_form_process.join()

        self.close_shmem()

        self.exchange_data = None
        self.close()
        print("main process finished")

    def close_shmem(self):

        self.mc_shmem.close()
        self.mc_shmem.unlink()
        self.mosaic_shmem.close()
        self.mosaic_shmem.unlink()
        self.epi_shmem.close()
        self.epi_shmem.unlink()
        # self.stat_shmem.close()
        # self.stat_shmem.unlink()
        self.proj_t_shmem.close()
        self.proj_t_shmem.unlink()
        self.proj_c_shmem.close()
        self.proj_c_shmem.unlink()
        self.proj_s_shmem.close()
        self.proj_s_shmem.unlink()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = OpenNFTManager()
    sys.exit(app.exec_())
