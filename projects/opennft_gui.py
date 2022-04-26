import time
import sys
import numpy as np
import pyqtgraph as pg
import multiprocessing as mp

from multiprocessing import shared_memory
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton
from loguru import logger

import opennft_proj
from opennft.config import config as con


class Window(QWidget):
    def __init__(self, *args, **kwargs):

        self.init_service_data()

        self._process = opennft_proj.OpenNFTCalc(self._service_data)

        self.nr_vol = self._service_data["nr_vol"]
        self.nr_rois = self._service_data["nr_rois"]

        self.init_shm()

        if con.use_gui:
            super().__init__(*args, **kwargs)

            loadUi('opennft.ui', self)

            self.mcPlot = self.createMcPlot(self.layoutPlot1)
            self.btnStart.clicked.connect(self.onStart)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.onTimer)
            self.show()
        else:
            self.onStart()

    def init_service_data(self):

        self._service_data = mp.Manager().dict()

        # flags "ready for display"
        # 0 - motion correction data
        self._service_data["data_ready_flags"] = False
        self._service_data["init"] = False
        self._service_data["nr_vol"] = 0
        self._service_data["nr_rois"] = 0

    def init_shm(self):

        base_array = np.zeros((self.nr_vol, 6), dtype=np.float32)
        self.mc_shm = shared_memory.SharedMemory(create=True, size=base_array.nbytes, name=con.shm_file_names[0])
        self.mc_data = np.ndarray(shape=base_array.shape, dtype=base_array.dtype, buffer=self.mc_shm.buf)

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
        if not self._process.is_alive():
            print("main starting process")
            self._process.start()
            if con.use_gui:
                self.timer.start(30)
        else:
            pass

    def onTimer(self):
        if self._service_data["data_ready_flags"]:
            self.drawMcPlots(self.mcPlot, self.mc_data)
            self._service_data["data_ready_flags"] = False

    def closeEvent(self, event):
        if self._process.is_alive():
            self._process.join()
        self.mc_shm.close()
        self.mc_shm.unlink()
        self._service_data = None
        self.close()
        print("main process finished")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())
