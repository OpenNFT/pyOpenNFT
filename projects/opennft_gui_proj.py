import enum
import re
import time
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import multiprocessing as mp

from multiprocessing import shared_memory
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer, QSettings
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMenu
from loguru import logger

import opennft_console_proj
from opennft import mosaicview, projview, mapimagewidget, volviewformation
from opennft.config import config as con
from opennft import constants as cons
from opennft.mrvol import MrVol


class ImageViewMode(str, enum.Enum):
    mosaic = 'mosaic'
    orthview_anat = 'bgAnat'
    orthview_epi = 'bgEPI'


class OpenNFTManager(QWidget):

    # --------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):

        self.init_exchange_data()

        self.setting_file_name = Path(__file__).absolute().resolve().parent
        self.tcp_data = dict.fromkeys(["use_tcp_data", "tcp_data_ip", "tcp_data_port"])

        self.orthViewInitialize = False

        if con.use_gui:
            super().__init__(*args, **kwargs)

            loadUi('opennft.ui', self)
            self.plotBgColor = (255, 255, 255)

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
            self.orthView.cursorPositionChanged.connect(self.onChangeOrthViewCursorPosition)

            self.mcPlot = self.createMcPlot(self.layoutPlot1)
            self.rawRoiPlot, self.procRoiPlot, self.normRoiPlot = self.createRoiPlots()
            self.tsTimer = QTimer(self)
            self.mosaicTimer = QTimer(self)
            self.orthViewTimer = QTimer(self)
            self.tsTimer.timeout.connect(self.onCheckTimeSeriesUpdated)
            self.mosaicTimer.timeout.connect(self.onCheckMosaicViewUpdated)
            self.orthViewTimer.timeout.connect(self.onCheckOrthViewUpdated)

            self.currentCursorPos = (129, 95)
            self.currentProjection = projview.ProjectionType.coronal

            self.pbMoreParameters.setChecked(False)

            self.leFirstFile.textChanged.connect(lambda: self.textChangedDual(self.leFirstFile, self.leFirstFile2))
            self.leFirstFile2.textChanged.connect(lambda: self.textChangedDual(self.leFirstFile2, self.leFirstFile))

            self.btnChooseSetFile.clicked.connect(self.onChooseSetFile)
            self.btnChooseSetFile2.clicked.connect(self.onChooseSetFile)
            self.pbMoreParameters.toggled.connect(self.onShowMoreParameters)

            self.pos_map_thresholds_widget.thresholds_manually_changed.connect(self.onInteractWithMapImage)
            self.neg_map_thresholds_widget.thresholds_manually_changed.connect(self.onInteractWithMapImage)

            self.exchange_data["pos_thresholds"] = self.pos_map_thresholds_widget.get_thresholds()
            self.exchange_data["neg_thresholds"] = self.neg_map_thresholds_widget.get_thresholds()

            self.posMapCheckBox.toggled.connect(self.onChangePosMapVisible)
            self.negMapCheckBox.toggled.connect(self.onChangeNegMapVisible)

            self.sliderMapsAlpha.valueChanged.connect(lambda v: self.mosaicImageView.set_pos_map_opacity(v / 100.0))
            self.sliderMapsAlpha.valueChanged.connect(lambda v: self.mosaicImageView.set_neg_map_opacity(v / 100.0))
            self.sliderMapsAlpha.valueChanged.connect(lambda v: self.orthView.set_pos_map_opacity(v / 100.0))
            self.sliderMapsAlpha.valueChanged.connect(lambda v: self.orthView.set_neg_map_opacity(v / 100.0))

            self.btnSetup.clicked.connect(self.setup)
            self.btnStart.clicked.connect(self.start)
            self.btnStop.clicked.connect(self.stop)

            self.init = False
            self.show()
        else:
            self.setup()
            self.start()

    # --------------------------------------------------------------------------
    def init_exchange_data(self):

        self.exchange_data = mp.Manager().dict()

        self.exchange_data["set_file"] = ""

        self.exchange_data["data_ready_flag"] = False
        self.exchange_data["init"] = False
        self.exchange_data["nr_vol"] = 0
        self.exchange_data["nr_rois"] = 0
        self.exchange_data["vol_dim"] = 0
        self.exchange_data["mosaic_dim"] = 0
        self.exchange_data["is_ROI"] = con.use_roi
        self.exchange_data["is_rtqa"] = con.use_rtqa
        self.exchange_data["vol_mat"] = None
        self.exchange_data["is_stopped"] = False
        self.exchange_data["ready_to_form"] = False
        self.exchange_data["view_mode"] = 'mosaic'
        self.exchange_data["done_mosaic_templ"] = False
        self.exchange_data["done_mosaic_overlay"] = False
        self.exchange_data["done_orth"] = False
        self.exchange_data["overlay_ready"] = False
        self.exchange_data["iter_norm_number"] = 0
        self.exchange_data["elapsed_time"] = 0
        self.exchange_data["ROI_t"] = np.zeros((1, 1))
        self.exchange_data["ROI_c"] = np.zeros((1, 1))
        self.exchange_data["ROI_s"] = np.zeros((1, 1))
        self.exchange_data["auto_thr_pos"] = True
        self.exchange_data["auto_thr_neg"] = False
        self.exchange_data["proj_dims"] = None
        self.exchange_data["is_neg"] = False
        self.exchange_data["bg_type"] = "bgEPI"
        self.exchange_data["cursor_pos"] = (129, 95)
        self.exchange_data["flags_planes"] = projview.ProjectionType.coronal.value

    # --------------------------------------------------------------------------
    def init_shmem(self):

        mc_array = np.zeros((self.nr_vol, 6), dtype=np.float32)
        self.mc_shmem = shared_memory.SharedMemory(create=True, size=mc_array.nbytes, name=con.shmem_file_names[0])
        self.mc_data = np.ndarray(shape=mc_array.shape, dtype=mc_array.dtype, buffer=self.mc_shmem.buf)

        time_series_array = np.zeros((5, self.nr_vol, self.nr_rois), dtype=np.float32)
        self.ts_shmem = shared_memory.SharedMemory(create=True, size=time_series_array.nbytes,
                                                   name=con.shmem_file_names[8])
        self.ts_data = np.ndarray(shape=time_series_array.shape, dtype=time_series_array.dtype,
                                  buffer=self.ts_shmem.buf)

        nfb_array = np.zeros((1, self.nr_vol), dtype=np.float32)
        self.nfb_shmem = shared_memory.SharedMemory(create=True, size=nfb_array.nbytes, name=con.shmem_file_names[9])
        self.nfb_data = np.ndarray(shape=nfb_array.shape, dtype=nfb_array.dtype, buffer=self.nfb_shmem.buf)

        mosaic_array = np.zeros(self.mosaic_dim + (9,), dtype=np.float32)
        self.mosaic_shmem = shared_memory.SharedMemory(create=True, size=mosaic_array.nbytes * 9,
                                                       name=con.shmem_file_names[1])
        self.mosaic_data = np.ndarray(shape=mosaic_array.shape, dtype=mosaic_array.dtype, buffer=self.mosaic_shmem.buf)

        vol_array = np.zeros(self.exchange_data["vol_dim"], dtype=np.float32)
        self.epi_shmem = shared_memory.SharedMemory(create=True, size=vol_array.nbytes, name=con.shmem_file_names[2])
        self.epi_data = np.ndarray(shape=vol_array.shape, dtype=vol_array.dtype, buffer=self.epi_shmem.buf, order='F')
        self.epi_data[:, :, :] = self._core_process.session.reference_vol.volume

        stat_dim = tuple(self.exchange_data["vol_dim"]) + (2,)
        stat_array = np.zeros(stat_dim, dtype=np.float32)
        self.stat_shmem = shared_memory.SharedMemory(create=True, size=stat_array.nbytes, name=con.shmem_file_names[3])
        self.stat_data = np.ndarray(shape=stat_array.shape, dtype=stat_array.dtype, buffer=self.stat_shmem.buf,
                                    order='F')

        dims = self.exchange_data["proj_dims"]
        proj_array = np.zeros((dims[1], dims[0], 9), dtype=np.float32)
        self.proj_t_shmem = shared_memory.SharedMemory(create=True, size=proj_array.nbytes,
                                                       name=con.shmem_file_names[5])
        self.proj_t = np.ndarray(shape=(dims[1], dims[0], 9), dtype=np.float32, buffer=self.proj_t_shmem.buf)

        proj_array = np.zeros((dims[2], dims[0], 9), dtype=np.float32)
        self.proj_c_shmem = shared_memory.SharedMemory(create=True, size=proj_array.nbytes,
                                                       name=con.shmem_file_names[6])
        self.proj_c = np.ndarray(shape=(dims[2], dims[0], 9), dtype=np.float32, buffer=self.proj_c_shmem.buf)

        proj_array = np.zeros((dims[2], dims[1], 9), dtype=np.float32)
        self.proj_s_shmem = shared_memory.SharedMemory(create=True, size=proj_array.nbytes,
                                                       name=con.shmem_file_names[7])
        self.proj_s = np.ndarray(shape=(dims[2], dims[1], 9), dtype=np.float32, buffer=self.proj_s_shmem.buf)

    # --------------------------------------------------------------------------
    def view_form_init(self):

        if con.use_roi:
            ROI_vols = np.zeros(((self.nr_rois,)+tuple(self.session.rois[0].dim)))
            ROI_mats = np.zeros((self.nr_rois, 4, 4))
            for i_roi in range(self.nr_rois):
                ROI_vols[i_roi, :, :, :] = self.session.rois[i_roi].volume
                ROI_mats[i_roi, :, :] = self.session.rois[i_roi].mat
        else:
            ROI_vols = []
            ROI_mats = []

        self._view_form_process = volviewformation.VolViewFormation(self.exchange_data, ROI_vols, ROI_mats)

    # --------------------------------------------------------------------------
    def textChangedDual(self, leFrom, leTo):
        pos = leTo.cursorPosition()
        leTo.setText(leFrom.text())
        leTo.setCursorPosition(pos)

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    def createRoiPlots(self):
        rawroiplot = pg.PlotWidget(self)
        self.layoutPlot2.addWidget(rawroiplot)

        p = rawroiplot.getPlotItem()
        p.setTitle('Raw ROI', size='')
        p.setLabel('left', "Amplitude [a.u.]")
        p.setMenuEnabled(enableMenu=False)
        p.setMouseEnabled(x=False, y=False)
        p.installEventFilter(self)

        procroiplot = pg.PlotWidget(self)
        self.layoutPlot3.addWidget(procroiplot)

        p = procroiplot.getPlotItem()
        p.setTitle('Proc ROI', size='')
        p.setLabel('left', "Amplitude [a.u.]")
        p.setMenuEnabled(enableMenu=False)
        p.setMouseEnabled(x=False, y=False)
        p.installEventFilter(self)

        normroiplot = pg.PlotWidget(self)
        self.layoutPlotMain.addWidget(normroiplot)

        p = normroiplot.getPlotItem()
        p.setTitle('Norm ROI', size='')
        p.setLabel('left', "Amplitude [a.u.]")
        p.setMenuEnabled(enableMenu=False)
        p.setMouseEnabled(x=False, y=False)
        p.installEventFilter(self)

        plots = (
            rawroiplot,
            procroiplot,
            normroiplot
        )

        for p in plots:
            p.setBackground(self.plotBgColor)

        return plots

    # --------------------------------------------------------------------------
    def createMusterInfo(self):
        # TODO: More general way to use any protocol
        tmpCond = list()
        nrCond = list()
        for c in self.session.offsets:
            tmpCond.append(c)
            nrCond.append(tmpCond[-1].shape[0])

        if not ('BAS' in self.session.prot_names):  # implicit baseline
            # self.P['ProtCond'][0] - 0 is for Baseline indexes
            tmpCond.insert(0, np.array([[t[n] for t in self.session.prot_cond[0]] for n in (0, -1)]).T)
            nrCond.insert(0, tmpCond[0].shape[0])

        c = 1
        for c in range(len(tmpCond), 4):  # placeholders
            tmpCond.append(np.array([(0, 0), (0, 0)]))
            nrCond.append(tmpCond[-1].shape[0])

        if self.config.prot == 'InterBlock':
            blockLength = tmpCond[0][0][1] - tmpCond[0][0][0] + c
        else:
            blockLength = 0
            for condNumber in range(len(tmpCond)):
                blockLength += tmpCond[condNumber][0][1] - tmpCond[condNumber][0][0]
            blockLength += c

        def removeIntervals(data, remData):
            dfs = []

            for n1, n2 in zip(data[:-1], data[1:]):
                df = n2[0] - n1[1] - blockLength - 1
                dfs.append(df)

            dfs = np.cumsum(dfs)

            idx = []
            last = 0

            for i, n in enumerate(dfs):
                if n > last:
                    idx.append(i + 1)
                    last = n

            for i, r in zip(idx, remData[:-1]):
                sz = (r[1] - r[0] + 1)
                data[i:, 0] -= sz
                data[i:, 1] -= sz

        if self.config.prot == 'InterBlock':
            remCond = []

            for a, b in zip(tmpCond[2], tmpCond[3]):
                remCond.append((a[0], b[1]))

            removeIntervals(tmpCond[0], remCond)
            removeIntervals(tmpCond[1], remCond)

        # To break drawMusterPlot() at given length of conditions,
        # i.e., to avoid plotting some of them as for DCM feedback type
        condTotal = 2 if self.config.prot == 'InterBlock' else len(tmpCond)

        tmpCondStr = ['tmpCond{:d}'.format(x + 1) for x in range(condTotal)]
        nrCondStr = ['nrCond{:d}'.format(x + 1) for x in range(condTotal)]
        self.musterInfo = dict.fromkeys(tmpCondStr + nrCondStr)
        self.musterInfo['condTotal'] = condTotal
        for condNumber in range(condTotal):
            self.musterInfo[tmpCondStr[condNumber]] = tmpCond[condNumber]
            self.musterInfo[nrCondStr[condNumber]] = nrCond[condNumber]
        self.musterInfo['blockLength'] = blockLength

    # --------------------------------------------------------------------------
    def setupMcPlots(self):
        mctrrot = self.mcPlot.getPlotItem()
        self.basicSetupPlot(mctrrot)

    # --------------------------------------------------------------------------
    def setupRoiPlots(self):
        self.makeRoiPlotLegend()

        rawTimeSeries = self.rawRoiPlot.getPlotItem()
        proc = self.procRoiPlot.getPlotItem()
        norm = self.normRoiPlot.getPlotItem()

        rawTimeSeries.clear()
        proc.clear()
        norm.clear()

        # TODO: For autoRTQA mode
        if True:
            grid = True
        else:
            grid = False

        self.basicSetupPlot(rawTimeSeries, grid)
        self.basicSetupPlot(proc, grid)
        self.basicSetupPlot(norm, grid)

        self.drawMusterPlot(rawTimeSeries)
        self.drawMusterPlot(proc)
        self.drawMusterPlot(norm)
        rawTimeSeries.setYRange(-1, 1, padding=0.0)
        proc.setYRange(-1, 1, padding=0.0)
        norm.setYRange(-1, 1, padding=0.0)

    # --------------------------------------------------------------------------
    def basicSetupPlot(self, plotitem, grid=True):
        # For autoRTQA mode
        if True:
            lastInds = np.zeros((self.musterInfo['condTotal'],))
            for i in range(self.musterInfo['condTotal']):
                lastInds[i] = self.musterInfo['tmpCond' + str(i + 1)][-1][1]
            xmax = max(lastInds)
        else:
            xmax = self.nr_vol

        plotitem.disableAutoRange(axis=pg.ViewBox.XAxis)
        plotitem.setXRange(1, xmax, padding=0.0)
        plotitem.showGrid(x=grid, y=grid, alpha=cons.PLOT_GRID_ALPHA)

    # --------------------------------------------------------------------------
    def makeRoiPlotLegend(self):
        roiNames = []

        for roiName in self.exchange_data["roi_names"]:
            roiName = Path(roiName).stem
            if len(roiName) > cons.MAX_ROI_NAME_LENGTH:
                roiName = roiName[:2] + '..' + roiName[-2:]
            roiNames.append(roiName)

        self.labelPlotLegend.setText('')
        legendText = '<html><head/><body><p>'

        numRoi = int(self.nr_rois)

        for i, n, c in zip(range(1, numRoi + 1), roiNames, cons.ROI_PLOT_COLORS):
            cname = pg.mkPen(color=c).color().name()
            legendText += (
                    '<span style="font-weight:600;color:{};">'.format(cname)
                    + 'ROI_{} {}</span>, '.format(i, n))

        legendText += (
            '<span style="font-weight:600;color:k;">Operation: {}</span>'.format(self.config.roi_anat_operation))
        legendText += '</p></body></html>'

        self.labelPlotLegend.setText(legendText)

    # --------------------------------------------------------------------------
    def computeMusterPlotData(self, ylim):
        singleY = np.array([ylim[0], ylim[1], ylim[1], ylim[0]])

        def computeConds(nrCond, tmpCond):
            xCond = np.zeros(nrCond * 4, dtype=np.float64)
            yCond = np.zeros(nrCond * 4, dtype=np.float64)

            for k in range(nrCond):
                i = slice(k * 4, (k + 1) * 4)

                xCond[i] = np.array([
                    tmpCond[k][0] - 1,
                    tmpCond[k][0] - 1,
                    tmpCond[k][1],
                    tmpCond[k][1],
                ])

                yCond[i] = singleY

            return xCond, yCond

        for cond in range(self.musterInfo['condTotal']):
            xCond, yCond = computeConds(self.musterInfo['nrCond' + str(cond + 1)],
                                        self.musterInfo['tmpCond' + str(cond + 1)])
            self.musterInfo['xCond' + str(cond + 1)] = xCond
            self.musterInfo['yCond' + str(cond + 1)] = yCond

    # --------------------------------------------------------------------------
    def drawMusterPlot(self, plotitem: pg.PlotItem):
        ylim = cons.MUSTER_Y_LIMITS

        # For autoRTQA mode
        if True:
            self.computeMusterPlotData(ylim)
            muster = []

            for i in range(self.musterInfo['condTotal']):
                muster.append(
                    plotitem.plot(x=self.musterInfo['xCond' + str(i + 1)],
                                  y=self.musterInfo['yCond' + str(i + 1)],
                                  fillLevel=ylim[0],
                                  pen=cons.MUSTER_PEN_COLORS[i],
                                  brush=cons.MUSTER_BRUSH_COLORS[i])
                )

        else:
            muster = [
                plotitem.plot(x=[1, self.nr_vol],
                              y=[-1000, 1000],
                              fillLevel=ylim[0],
                              pen=cons.MUSTER_PEN_COLORS[9],
                              brush=cons.MUSTER_BRUSH_COLORS[9])
            ]

        return muster

    # --------------------------------------------------------------------------
    def drawMcPlots(self, init, mcPlot, data):

        mctrrot = mcPlot.getPlotItem()

        if init:
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

    # --------------------------------------------------------------------------
    def drawRoiPlots(self, init, data, iter):

        data_raw = np.array(data[0, :, self.selected_roi].squeeze(), ndmin=2)
        data_proc = np.array(data[1, :, self.selected_roi].squeeze(), ndmin=2)
        data_norm = np.array(data[2, :, self.selected_roi].squeeze(), ndmin=2)
        data_pos = np.array(data[3:5, :, self.selected_roi], ndmin=2)

        if self.config.plot_feedback:
            if iter == 1:
                data_norm = np.hstack((data_norm, self.nfb_data[:, 0:iter]))
            else:
                data_norm = np.vstack((data_norm, self.nfb_data[:, 0:iter]))

        self.drawGivenRoiPlot(init, self.rawRoiPlot, data_raw)
        self.drawGivenRoiPlot(init, self.procRoiPlot, data_proc, data_pos)
        self.drawGivenRoiPlot(init, self.normRoiPlot, data_norm)

    # --------------------------------------------------------------------------
    def drawGivenRoiPlot(self, init, plotwidget: pg.PlotWidget, data, data_pos=None):
        plotitem = plotwidget.getPlotItem()

        sz, l = data.shape

        if init:

            plotitem.enableAutoRange(enable=True, x=False, y=True)

            plotitem.clear()
            muster = self.drawMusterPlot(plotitem)

            plots = []

            plot_colors = np.array(cons.ROI_PLOT_COLORS)[self.selected_roi]
            if self.config.max_feedback_val:
                plot_colors = np.append(plot_colors, cons.ROI_PLOT_COLORS[int(self.nr_rois)])
            for i, c in zip(range(sz), plot_colors):
                pen = pg.mkPen(color=c, width=cons.ROI_PLOT_WIDTH)
                p = plotitem.plot(pen=pen)
                plots.append(p)

            self.drawGivenRoiPlot.__dict__[plotitem] = plots, muster

        x = np.arange(1, l + 1, dtype=np.float64)

        for p, y in zip(self.drawGivenRoiPlot.__dict__[plotitem][0], data):
            p.setData(x=x, y=np.array(y))

        if self.config.prot != 'InterBlock':
            if plotwidget == self.procRoiPlot:
                posMin = np.array(data_pos[0, :, :].squeeze(), ndmin=2).T
                posMax = np.array(data_pos[1, :, :].squeeze(), ndmin=2).T
                inds = list(self.selected_roi)
                inds.append(len(posMin) - 1)
                posMin = posMin[inds]
                posMax = posMax[inds]

                self.drawMinMaxProcRoiPlot(
                    init, posMin, posMax)

        items = plotitem.listDataItems()

        for m in self.drawGivenRoiPlot.__dict__[plotitem][1]:
            items.remove(m)

        plotitem.autoRange(items=items)

        # For autoRTQA
        if False:
            grid = True
        else:
            grid = False
        self.basicSetupPlot(plotitem, grid)

    # --------------------------------------------------------------------------
    def drawMinMaxProcRoiPlot(self, init, posMin, posMax):
        plotitem = self.procRoiPlot.getPlotItem()
        sz = posMin.shape[0] + 1
        l = posMin.shape[1]

        if init:
            plotsMin = []
            plotsMax = []

            plot_colors = np.array(cons.ROI_PLOT_COLORS)
            plot_colors = np.append(plot_colors[self.selected_roi], plot_colors[-1])
            for i, c in zip(range(sz), plot_colors):
                plotsMin.append(plotitem.plot(pen=pg.mkPen(
                    color=c, width=cons.ROI_PLOT_WIDTH)))
                plotsMax.append(plotitem.plot(pen=pg.mkPen(
                    color=c, width=cons.ROI_PLOT_WIDTH)))

            self.drawMinMaxProcRoiPlot.__dict__['posMin'] = plotsMin
            self.drawMinMaxProcRoiPlot.__dict__['posMax'] = plotsMax

        x = np.arange(1, l + 1, dtype=np.float64)

        for pmi, mi, pma, ma in zip(
                self.drawMinMaxProcRoiPlot.__dict__['posMin'], posMin,
                self.drawMinMaxProcRoiPlot.__dict__['posMax'], posMax):
            mi = np.array(mi, ndmin=1)
            ma = np.array(ma, ndmin=1)
            pmi.setData(x=x, y=mi)
            pma.setData(x=x, y=ma)

    # --------------------------------------------------------------------------
    def setup(self):

        self._core_process = opennft_console_proj.OpenNFTCoreProj(self.exchange_data)

        self.config = self._core_process.config
        self.session = self._core_process.session
        self.nr_vol = self.exchange_data["nr_vol"]
        self.nr_rois = self.exchange_data["nr_rois"]
        self.vol_dim = self.exchange_data["vol_dim"]
        self.mosaic_dim = self.exchange_data["mosaic_dim"]
        self.overlay_dim = self.exchange_data["mosaic_dim"] + (4,)

        self.roi_dict = dict()
        self.selected_roi = []
        roi_menu = QMenu()
        roi_menu.triggered.connect(self.onRoiChecked)
        self.roiSelectorBtn.setMenu(roi_menu)
        nrROIs = int(self.nr_rois)
        for i in range(nrROIs):
            # if self.P['isRTQA'] and i + 1 == nrROIs:
            #     roi = 'Whole brain ROI'
            # else:
            roi = 'ROI_{}'.format(i + 1)
            roi_action = roi_menu.addAction(roi)
            roi_action.setCheckable(True)
            if not (self.exchange_data["is_rtqa"] and i + 1 == nrROIs):
                roi_action.setChecked(True)
                self.roi_dict[roi] = True
                self.selected_roi.append(i)

        action = roi_menu.addAction("All")
        action.setCheckable(False)

        action = roi_menu.addAction("None")
        action.setCheckable(False)

        self.roiSelectorBtn.setEnabled(True)

        self.view_form_init()

        self.init_shmem()
        self._view_form_process.start()

        logger.info("  Setup plots...")
        self.createMusterInfo()

        self.setupRoiPlots()
        self.setupMcPlots()

        self.btnStart.setEnabled(True)

    # --------------------------------------------------------------------------
    def start(self):

        if not self._core_process.is_alive():

            self.cbImageViewMode.setEnabled(True)
            self.pos_map_thresholds_widget.setEnabled(True)
            self.neg_map_thresholds_widget.setEnabled(True)
            self.btnSetup.setEnabled(False)
            self.btnStart.setEnabled(False)
            self.btnStop.setEnabled(True)
            self.pbMoreParameters.setChecked(False)
            self.init = True

            print("main starting process")
            self._core_process.start()
            if con.use_gui:
                self.tsTimer.start(30)
                self.mosaicTimer.start(30)
                self.orthViewTimer.start(30)
        else:
            pass

    # --------------------------------------------------------------------------
    def stop(self):

        if self._core_process.is_alive():
            self._core_process.terminate()

        self.exchange_data["is_stopped"] = True
        # if self.windowRTQA:
        #     if not self.rtqa_input is None:
        #         self.rtqa_input["is_stopped"] = True
        #     self.eng.workspace['rtQA_python'] = self.calc_rtqa.dataPacking()
        self.btnStop.setEnabled(False)

        # if 'isAutoRTQA' in self.P and not self.P['isAutoRTQA']:
        self.btnStart.setEnabled(False)
        self.btnSetup.setEnabled(True)
        self.btnPlugins.setEnabled(True)
        # else:
        #     self.btnStart.setEnabled(True)
        #     self.btnSetup.setEnabled(False)
        #     self.btnPlugins.setEnabled(False)

        if con.use_sleep_in_stop:
            time.sleep(2)

        self.reset_done = False

        # if config.USE_MRPULSE and hasattr(self, 'mrPulses'):
        #     np_arr = mrpulse.toNpData(self.mrPulses)
        #     self.pulseProc.terminate()

        # if self.iteration > 1 and self.P.get('nfbDataFolder'):
        #     path = Path(self.P['nfbDataFolder'])
        #     fname = path / ('TimeVectors_' + str(self.P['NFRunNr']).zfill(2) + '.txt')
        #     self.recorder.savetxt(str(fname))

        # if self.fFinNFB:
        #     for i in range(len(self.plugins)):
        #         self.plugins[i].finalize()
        #     self.finalizeUdpSender()
        #     self.nfbFinStarted = self.eng.nfbSave(self.iteration, nargout=0, background=True)
        #     self.fFinNFB = False

        # if self.recorder.records.shape[0] > 2:
        #     if self.recorder.records[0, erd.Times.d0] > 0:
        #         logger.info("Average elapsed time: {:.4f} s".format(
        #             np.sum(self.recorder.records[1:, erd.Times.d0]) / self.recorder.records[0, erd.Times.d0]))

        logger.info('Finished.')

    # --------------------------------------------------------------------------
    def reset(self):
        self.exchange_data["is_stopped"] = True

        if self._core_process.is_alive():
            self._core_process.terminate()
        if self._view_form_process.is_alive():
            self._view_form_process.terminate()
        self._core_process = None
        self._view_form_process = None

        self.mcPlot.getPlotItem().clear()
        self.procRoiPlot.getPlotItem().clear()
        self.rawRoiPlot.getPlotItem().clear()
        self.normRoiPlot.getPlotItem().clear()

        self.pos_map_thresholds_widget.reset()
        self.neg_map_thresholds_widget.reset()

        self.mosaicImageView.clear()
        self.orthView.clear()

        self.reset_done = True

    # --------------------------------------------------------------------------
    def onChooseSetFile(self):
        if con.donot_use_qfile_native_dialog:
            fname = QFileDialog.getOpenFileName(
                self, "Select 'SET File'", str(self.setting_file_name), 'ini files (*.ini)',
                options=QFileDialog.DontUseNativeDialog)[0]
        else:
            fname = QFileDialog.getOpenFileName(
                self, "Select 'SET File'", str(self.setting_file_name), 'ini files (*.ini)')[0]

        fname = str(Path(fname))
        self.chooseSetFile(fname)

    # --------------------------------------------------------------------------
    def onShowMoreParameters(self, flag: bool):
        self.stackedWidgetMain.setCurrentIndex(int(flag))

    # --------------------------------------------------------------------------
    def chooseSetFile(self, fname):
        if not fname:
            return

        if not Path(fname).is_file():
            return

        self.setting_file_name = fname

        self.leSetFile.setText(fname)
        self.exchange_data["set_file"] = fname

        self.settings = QSettings(fname, QSettings.IniFormat, self)
        self.loadSettingsFromSetFile()

        self.is_set_file_chosen = True
        self.btnSetup.setEnabled(True)

    # --------------------------------------------------------------------------
    def loadSettingsFromSetFile(self):

        # --- top ---
        self.leProtocolFile.setText(self.settings.value('StimulationProtocol', ''))
        self.leWorkFolder.setText(self.settings.value('WorkFolder', ''))
        self.leWatchFolder.setText(self.settings.value('WatchFolder', ''))
        if (self.settings.value('Type', '')) == 'DCM':
            self.leRoiAnatFolder.setText(self.settings.value('RoiAnatFolder', ''))
        else:
            self.leRoiAnatFolder.setText(self.settings.value('RoiFilesFolder', ''))
        self.leRoiAnatOperation.setText(self.settings.value('RoiAnatOperation', 'mean(norm_percValues)'))
        self.leRoiGroupFolder.setText(self.settings.value('RoiGroupFolder', ''))
        self.leStructBgFile.setText(self.settings.value('StructBgFile', ''))
        self.leMCTempl.setText(self.settings.value('MCTempl', ''))
        if (self.settings.value('Prot', '')) == 'ContTask':
            self.leTaskFolder.setText(self.settings.value('TaskFolder', ''))

        # --- middle ---
        self.leProjName.setText(self.settings.value('ProjectName', ''))
        self.leSubjectID.setText(self.settings.value('SubjectID', ''))
        self.leFirstFile.setText(self.settings.value('FirstFileNameTxt', '001_{Image Series No:06}_{#:06}.dcm'))
        self.sbNFRunNr.setValue(int(self.settings.value('NFRunNr', '1')))
        self.sbImgSerNr.setValue(int(self.settings.value('ImgSerNr', '1')))
        self.sbVolumesNr.setValue(int(self.settings.value('NrOfVolumes')))
        self.sbSlicesNr.setValue(int(self.settings.value('NrOfSlices')))
        self.sbTR.setValue(int(self.settings.value('TR')))
        self.sbSkipVol.setValue(int(self.settings.value('nrSkipVol')))
        self.sbMatrixSizeX.setValue(int(self.settings.value('MatrixSizeX')))
        self.sbMatrixSizeY.setValue(int(self.settings.value('MatrixSizeY')))

        # --- bottom left ---
        self.cbOfflineMode.setChecked(str(self.settings.value('OfflineMode', 'true')).lower() == 'true')

        if self.settings.value('UseTCPData', None) is None:
            logger.warning('Upgrade settings format from version 1.0.rc0')

        self.cbUseTCPData.setChecked(str(self.settings.value('UseTCPData', 'false')).lower() == 'true')
        if self.cbUseTCPData.isChecked():
            self.leTCPDataIP.setText(self.settings.value('TCPDataIP', ''))
            self.leTCPDataPort.setText(str(self.settings.value('TCPDataPort', '')))

        self.leMaxFeedbackVal.setText(str(self.settings.value('MaxFeedbackVal', '100')))  # FixMe
        self.leMinFeedbackVal.setText(str(self.settings.value('MinFeedbackVal', '-100')))
        self.sbFeedbackValDec.setValue(int(self.settings.value('FeedbackValDec', '0')))  # FixMe
        self.cbNegFeedback.setChecked(str(self.settings.value('NegFeedback', 'false')).lower() == 'true')
        self.cbFeedbackPlot.setChecked(str(self.settings.value('PlotFeedback', 'true')).lower() == 'true')

        self.leShamFile.setText(self.settings.value('ShamFile', ''))

        self.cbUsePTB.setChecked(str(self.settings.value('UsePTB', 'false')).lower() == 'true')
        # if not config.USE_PTB_HELPER:
        #     self.cbUsePTB.setChecked(False)
        #     self.cbUsePTB.setEnabled(False)

        self.cbScreenId.setCurrentIndex(int(self.settings.value('DisplayFeedbackScreenID', 0)))
        self.cbDisplayFeedbackFullscreen.setChecked(
            str(self.settings.value('DisplayFeedbackFullscreen')).lower() == 'true')

        self.cbUseUDPFeedback.setChecked(str(self.settings.value('UseUDPFeedback')).lower() == 'true')
        self.leUDPFeedbackIP.setText(self.settings.value('UDPFeedbackIP', ''))
        self.leUDPFeedbackPort.setText(str(self.settings.value('UDPFeedbackPort', '1234')))
        self.leUDPFeedbackControlChar.setText(str(self.settings.value('UDPFeedbackControlChar', '')))
        self.cbUDPSendCondition.setChecked(str(self.settings.value('UDPSendCondition')).lower() == 'true')

        # --- bottom right ---
        idx = self.cbDataType.findText(self.settings.value('DataType', 'DICOM'))
        if idx >= 0:
            self.cbDataType.setCurrentIndex(idx)
        self.cbgetMAT.setChecked(str(self.settings.value('GetMAT')).lower() == 'true')
        idx = self.cbProt.findText(self.settings.value('Prot', 'Inter'))
        if idx >= 0:
            self.cbProt.setCurrentIndex(idx)
        idx = self.cbType.findText(self.settings.value('Type', 'PSC'))
        if idx >= 0:
            self.cbType.setCurrentIndex(idx)

        # --- main viewer ---
        self.sbTargANG.setValue(float(self.settings.value('TargANG', 0)))
        self.sbTargRAD.setValue(float(self.settings.value('TargRAD', 0)))
        self.sbTargDIAM.setValue(float(self.settings.value('TargDIAM', 0.0)))
        self.leWeightsFile.setText(str(self.settings.value('WeightsFileName', '')))

        self.actualize()

    # --------------------------------------------------------------------------
    def actualize(self):
        logger.info("  Actualizing:")

        # --- top ---
        self.exchange_data['ProtocolFile'] = self.leProtocolFile.text()
        self.exchange_data['WorkFolder'] = self.leWorkFolder.text()
        self.exchange_data['WatchFolder'] = self.leWatchFolder.text()

        self.exchange_data['Type'] = self.cbType.currentText()
        self.exchange_data['RoiAnatOperation'] = self.leRoiAnatOperation.text()
        self.exchange_data['RoiGroupFolder'] = self.leRoiGroupFolder.text()
        self.exchange_data['StructBgFile'] = self.leStructBgFile.text()
        self.exchange_data['MCTempl'] = self.leMCTempl.text()

        # --- middle ---
        self.exchange_data['ProjectName'] = self.leProjName.text()
        self.exchange_data['SubjectID'] = self.leSubjectID.text()
        self.exchange_data['FirstFileNameTxt'] = self.leFirstFile.text()
        self.exchange_data['ImgSerNr'] = self.sbImgSerNr.value()
        self.exchange_data['NFRunNr'] = self.sbNFRunNr.value()

        self.exchange_data['NrOfVolumes'] = self.sbVolumesNr.value()
        self.exchange_data['NrOfSlices'] = self.sbSlicesNr.value()
        self.exchange_data['TR'] = self.sbTR.value()
        self.exchange_data['nrSkipVol'] = self.sbSkipVol.value()
        self.exchange_data['MatrixSizeX'] = self.sbMatrixSizeX.value()
        self.exchange_data['MatrixSizeY'] = self.sbMatrixSizeY.value()

        # --- bottom left ---
        self.exchange_data['UseTCPData'] = self.cbUseTCPData.isChecked()
        if self.exchange_data['UseTCPData']:
            self.exchange_data['TCPDataIP'] = self.leTCPDataIP.text()
            self.exchange_data['TCPDataPort'] = int(self.leTCPDataPort.text())
        self.exchange_data['DisplayFeedbackFullscreen'] = self.cbDisplayFeedbackFullscreen.isChecked()

        # --- bottom right ---
        self.exchange_data['DataType'] = str(self.cbDataType.currentText())
        self.exchange_data['Prot'] = str(self.cbProt.currentText())
        self.exchange_data['Type'] = str(self.cbType.currentText())
        self.exchange_data['isAutoRTQA'] = con.auto_rtqa
        self.exchange_data['isRTQA'] = con.use_rtqa
        self.exchange_data['isIGLM'] = con.use_iglm
        self.exchange_data['isDicomSiemensXA30'] = con.dicom_siemens_xa30
        self.exchange_data['useEPITemplate'] = con.use_epi_template
        self.exchange_data['isZeroPadding'] = con.is_zero_padding
        self.exchange_data['nrZeroPadVol'] = con.nr_zero_pad_vol

        if self.exchange_data['Prot'] == 'ContTask':
            self.exchange_data['TaskFolder'] = self.leTaskFolder.text()

        self.exchange_data['MaxFeedbackVal'] = float(self.leMaxFeedbackVal.text())
        self.exchange_data['MinFeedbackVal'] = float(self.leMinFeedbackVal.text())
        self.exchange_data['FeedbackValDec'] = self.sbFeedbackValDec.value()
        self.exchange_data['NegFeedback'] = self.cbNegFeedback.isChecked()
        self.exchange_data['PlotFeedback'] = self.cbFeedbackPlot.isChecked()

        self.exchange_data['ShamFile'] = self.leShamFile.text()

        # --- main viewer ---
        self.exchange_data['WeightsFileName'] = self.leWeightsFile.text()

        # Parsing FirstFileNameTxt template and replace it with variables ---
        fields = {
            'projectname': self.exchange_data['ProjectName'],
            'subjectid': self.exchange_data['SubjectID'],
            'imageseriesno': self.exchange_data['ImgSerNr'],
            'nfrunno': self.exchange_data['NFRunNr'],
            '#': 1
        }
        template = self.exchange_data['FirstFileNameTxt']
        template_elements = re.findall(r"\{([A-Za-z0-9_: ]+)\}", template)

        self.exchange_data['FirstFileName'] = self.exchange_data['FirstFileNameTxt']

        for template_element in template_elements:
            template = template.replace("{%s}" % template_element, "{%s}" % template_element.replace(" ", "").lower())

        self.exchange_data['FirstFileName'] = template.format(**fields)

        # Update GUI information
        self.leCurrentVolume.setText('%d' % self.exchange_data["iter_norm_number"])
        self.leFirstFilePath.setText(str(Path(self.exchange_data['WatchFolder'], self.exchange_data['FirstFileName'])))

        filePathStatus = ""
        if Path(self.exchange_data['WatchFolder']).is_dir():
            filePathStatus += "MRI Watch Folder exists. "
        else:
            filePathStatus += "MRI Watch Folder does not exists. "
        if Path(self.leFirstFilePath.text()).is_file():
            filePathStatus += "First file exists. "
        else:
            filePathStatus += "First file does not exist. "

        # if Path(self.P['WatchFolder'],self.P['FirstFileName']).is_dir()
        self.lbFilePathStatus.setText(filePathStatus)

        # Update settings file
        # --- top ---
        self.settings.setValue('StimulationProtocol', self.exchange_data['ProtocolFile'])
        self.settings.setValue('WorkFolder', self.exchange_data['WorkFolder'])
        self.settings.setValue('WatchFolder', self.exchange_data['WatchFolder'])
        self.settings.setValue('RoiAnatOperation', self.exchange_data['RoiAnatOperation'])
        self.settings.setValue('RoiGroupFolder', self.exchange_data['RoiGroupFolder'])
        self.settings.setValue('StructBgFile', self.exchange_data['StructBgFile'])
        self.settings.setValue('MCTempl', self.exchange_data['MCTempl'])

        if self.exchange_data['Prot'] == 'ContTask':
            self.settings.setValue('TaskFolder', self.exchange_data['TaskFolder'])

        # --- middle ---
        self.settings.setValue('ProjectName', self.exchange_data['ProjectName'])
        self.settings.setValue('SubjectID', self.exchange_data['SubjectID'])
        self.settings.setValue('ImgSerNr', self.exchange_data['ImgSerNr'])
        self.settings.setValue('NFRunNr', self.exchange_data['NFRunNr'])

        self.settings.setValue('FirstFileNameTxt', self.exchange_data['FirstFileNameTxt'])
        self.settings.setValue('FirstFileName', self.exchange_data['FirstFileName'])

        self.settings.setValue('NrOfVolumes', self.exchange_data['NrOfVolumes'])
        self.settings.setValue('NrOfSlices', self.exchange_data['NrOfSlices'])
        self.settings.setValue('TR', self.exchange_data['TR'])
        self.settings.setValue('nrSkipVol', self.exchange_data['nrSkipVol'])
        self.settings.setValue('MatrixSizeX', self.exchange_data['MatrixSizeX'])
        self.settings.setValue('MatrixSizeY', self.exchange_data['MatrixSizeY'])

        # --- bottom left ---
        self.settings.setValue('OfflineMode', self.cbOfflineMode.isChecked())
        self.settings.setValue('UseTCPData', self.cbUseTCPData.isChecked())
        if self.cbUseTCPData.isChecked():
            self.settings.setValue('TCPDataIP', self.leTCPDataIP.text())
            self.settings.setValue('TCPDataPort', int(self.leTCPDataPort.text()))

        self.settings.setValue('MaxFeedbackVal', self.exchange_data['MaxFeedbackVal'])
        self.settings.setValue('MinFeedbackVal', self.exchange_data['MinFeedbackVal'])
        self.settings.setValue('FeedbackValDec', self.exchange_data['FeedbackValDec'])
        self.settings.setValue('NegFeedback', self.exchange_data['NegFeedback'])
        self.settings.setValue('PlotFeedback', self.exchange_data['PlotFeedback'])

        self.settings.setValue('ShamFile', self.exchange_data['ShamFile'])

        self.settings.setValue('UsePTB', self.cbUsePTB.isChecked())
        self.settings.setValue('DisplayFeedbackScreenID', self.cbScreenId.currentIndex())
        self.settings.setValue('DisplayFeedbackFullscreen', self.cbDisplayFeedbackFullscreen.isChecked())

        self.settings.setValue('UseUDPFeedback', self.cbUseUDPFeedback.isChecked())
        self.settings.setValue('UDPFeedbackIP', self.leUDPFeedbackIP.text())
        self.settings.setValue('UDPFeedbackPort', int(self.leUDPFeedbackPort.text()))
        self.settings.setValue('UDPFeedbackControlChar', self.leUDPFeedbackControlChar.text())
        self.settings.setValue('UDPSendCondition', self.cbUDPSendCondition.isChecked())

        # --- bottom right ---
        self.settings.setValue('DataType', self.exchange_data['DataType'])
        self.settings.setValue('Prot', self.exchange_data['Prot'])
        self.settings.setValue('Type', self.exchange_data['Type'])

        self.settings.setValue('WeightsFileName', self.exchange_data['WeightsFileName'])

        # Update config
        self.tcp_data["use_tcp_data"] = self.cbUseTCPData.isChecked()
        if self.tcp_data["use_tcp_data"]:
            # TCP receiver settings
            self.tcp_data["tcp_data_ip"] = self.leTCPDataIP.text()
            self.tcp_data["tcp_data_port"] = int(self.leTCPDataPort.text())

        con.USE_SHAM = bool(len(self.exchange_data['ShamFile']))

        con.use_ptb = self.cbUsePTB.isChecked()

        self.use_udp_feedback = self.cbUseUDPFeedback.isChecked()
        if self.use_udp_feedback:
            # UDP sender settings
            self.udp_feedback_ip = self.leUDPFeedbackIP.text()
            self.udp_feedback_port = int(self.leUDPFeedbackPort.text())
            self.udp_feedback_controlchar = self.leUDPFeedbackControlChar.text()
            self.udp_send_condition = self.cbUDPSendCondition.isChecked()
        else:
            self.udp_send_condition = False

    # --------------------------------------------------------------------------
    def onRoiChecked(self, action):
        if action.text() == "All":
            actList = self.roiSelectorBtn.menu().actions()
            actList = actList[0:-2]
            for act in actList:
                act.setChecked(True)
                self.roi_dict[act.text()] = True
        elif action.text() == "None":
            actList = self.roiSelectorBtn.menu().actions()
            actList = actList[0:-2]
            for act in actList:
                act.setChecked(False)
                self.roi_dict[act.text()] = False
        else:
            self.roi_dict[action.text()] = action.isChecked()

        self.selected_roi = np.where(list(self.roi_dict.values()))[0]
        # if self.windowRTQA:
        #     self.rtqa_input["roi_checked"] = self.selected_roi

        self.drawRoiPlots(True)
        if self.isStopped:
            # self.windowRTQA.plotRTQA()
            self.updateOrthViewAsync()

    # --------------------------------------------------------------------------
    def onChangePosMapVisible(self):
        is_visible = self.posMapCheckBox.isChecked()

        self.mosaicImageView.set_pos_map_visible(is_visible)
        self.orthView.set_pos_map_visible(is_visible)

    # --------------------------------------------------------------------------
    def onChangeNegMapVisible(self):
        is_visible = self.negMapCheckBox.isChecked()

        self.exchange_data["is_neg"] = is_visible

        self.mosaicImageView.set_neg_map_visible(is_visible)
        self.orthView.set_neg_map_visible(is_visible)

    # --------------------------------------------------------------------------
    def onInteractWithMapImage(self):
        sender = self.sender()

        if sender is self.pos_map_thresholds_widget:
            self.pos_map_thresholds_widget.auto_thresholds = False
            self.exchange_data["auto_thr_pos"] = False
            self.exchange_data["pos_thresholds"] = self.pos_map_thresholds_widget.get_thresholds()

        if sender is self.neg_map_thresholds_widget:
            self.neg_map_thresholds_widget.auto_thresholds = False
            self.exchange_data["auto_thr_neg"] = False
            self.exchange_data["neg_thresholds"] = self.neg_map_thresholds_widget.get_thresholds()

        if self.exchange_data["view_mode"] == ImageViewMode.mosaic:
            self.updateMosaicViewAsync()
        else:
            self.updateOrthViewAsync()

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

        if self.cbImageViewMode.isEnabled():
            self.updateOrthViewAsync()
            self.onInteractWithMapImage()

    # --------------------------------------------------------------------------
    def updateMosaicViewAsync(self):

        # if self.windowRTQA:
        #     is_snr_map_created = self.rtqa_input["rtqa_vol_ready"]
        #     is_rtqa_volume = self.rtqa_output["show_vol"]
        # else:
        #     is_rtqa_volume = False
        #     is_snr_map_created = False

        is_rtqa_volume = False

        # if self.windowRTQA and self.view_form_input["is_rtqa"]:
        #     if self.rtqa_input["which_vol"] == 0:
        #         self.view_form_input["rtQA_volume"] = self.rtqa_output["snr_vol"]
        #     else:
        #         self.view_form_input["rtQA_volume"] = self.rtqa_output["cnr_vol"]

        if is_rtqa_volume:
            self.exchange_data["is_neg"] = False
        else:
            self.exchange_data["is_neg"] = self.negMapCheckBox.isChecked()

    # --------------------------------------------------------------------------
    def updateOrthViewAsync(self):

        if self.exchange_data["view_mode"] == ImageViewMode.orthview_epi:
            bgType = 'bgEPI'
        else:
            bgType = 'bgStruct'

        # is_rtqa_volume = self.rtqa_output["show_vol"] if self.windowRTQA else False
        is_rtqa_volume = False

        # if not self.view_form_input["ready"]:
        self.exchange_data["cursor_pos"] = self.currentCursorPos
        self.exchange_data["flags_planes"] = self.currentProjection.value
        self.exchange_data["bg_type"] = bgType
        self.exchange_data["is_rtqa"] = is_rtqa_volume
        # if self.windowRTQA:
        #     if self.rtqa_input["which_vol"] == 0:
        #         self.view_form_input["rtQA_volume"] = self.rtqa_output["snr_vol"]
        #     else:
        #         self.view_form_input["rtQA_volume"] = self.rtqa_output["cnr_vol"]
        if is_rtqa_volume:
            self.exchange_data["is_neg"] = False
        else:
            self.exchange_data["is_neg"] = self.negMapCheckBox.isChecked()

        # self.view_form_input["ready"] = True

    # --------------------------------------------------------------------------
    def onChangeOrthViewCursorPosition(self, pos, proj):
        self.currentCursorPos = pos
        self.currentProjection = proj

        logger.info('New cursor coords {} for proj "{}" have been received', pos, proj.name)
        self.updateOrthViewAsync()

    # --------------------------------------------------------------------------
    def onCheckTimeSeriesUpdated(self):

        if not (self.exchange_data is None) or not self.exchange_data["is_stopped"]:

            if self.exchange_data["data_ready_flag"]:

                iter_norm_number = self.exchange_data["iter_norm_number"]

                self.drawMcPlots(self.init, self.mcPlot, self.mc_data[:iter_norm_number,:])

                data = self.ts_data[:, :iter_norm_number, :]
                self.drawRoiPlots(self.init, data, iter_norm_number)

                # if self.init:
                #     self.init = False

                self.exchange_data["data_ready_flag"] = False

        self.leElapsedTime.setText('{:.4f}'.format(self.exchange_data["elapsed_time"]))
        self.leCurrentVolume.setText('%d' % self.exchange_data["iter_norm_number"])

    # --------------------------------------------------------------------------
    def onCheckMosaicViewUpdated(self):

        if self.exchange_data["view_mode"] == ImageViewMode.mosaic:

            if self.exchange_data["done_mosaic_templ"]:
                background_image = self.mosaic_data[:, :, 0].squeeze()
                if background_image.size > 0:
                    logger.info("Done mosaic template")
                    self.mosaicImageView.set_background_image(background_image)
                else:
                    return

                self.exchange_data["done_mosaic_templ"] = False

            # rtQA/Stat map display
            if self.exchange_data["done_mosaic_overlay"]:

                rgba_pos_map_image = self.mosaic_data[:, :, 1:5]
                pos_thr = self.exchange_data["pos_thresholds"]
                self.pos_map_thresholds_widget.set_thresholds(pos_thr)

                if rgba_pos_map_image is not None:
                    self.mosaicImageView.set_pos_map_image(rgba_pos_map_image)

                if not self.exchange_data["is_rtqa"] and self.negMapCheckBox.isChecked():

                    rgba_neg_map_image = self.mosaic_data[:, :, 5:9]
                    neg_thr = self.exchange_data["neg_thresholds"]
                    self.neg_map_thresholds_widget.set_thresholds(neg_thr)

                    if rgba_neg_map_image is not None:
                        self.mosaicImageView.set_neg_map_image(rgba_neg_map_image)

                self.exchange_data["done_mosaic_overlay"] = False

    # --------------------------------------------------------------------------
    def onCheckOrthViewUpdated(self):

        if self.exchange_data["view_mode"] != ImageViewMode.mosaic:

            rgba_pos_map_image = None
            rgba_neg_map_image = None

            for proj in projview.ProjectionType:

                if proj == projview.ProjectionType.transversal:

                    bg_image = self.proj_t[:, :, 0].squeeze()
                    rgba_pos_map_image = self.proj_t[:, :, 1:5]
                    if not self.exchange_data["is_rtqa"] and self.negMapCheckBox.isChecked():
                        rgba_neg_map_image = self.proj_t[:, :, 5:9]

                elif proj == projview.ProjectionType.sagittal:

                    bg_image = self.proj_s[:, :, 0].squeeze()
                    rgba_pos_map_image = self.proj_s[:, :, 1:5]
                    if not self.exchange_data["is_rtqa"] and self.negMapCheckBox.isChecked():
                        rgba_neg_map_image = self.proj_s[:, :, 5:9]

                elif proj == projview.ProjectionType.coronal:

                    bg_image = self.proj_c[:, :, 0].squeeze()
                    rgba_pos_map_image = self.proj_c[:, :, 1:5]
                    if not self.exchange_data["is_rtqa"] and self.negMapCheckBox.isChecked():
                        rgba_neg_map_image = self.proj_c[:, :, 5:9]

                self.orthView.set_background_image(proj, bg_image)
                if rgba_pos_map_image is not None:
                    self.orthView.set_pos_map_image(proj, rgba_pos_map_image)
                if rgba_neg_map_image is not None:
                    self.orthView.set_neg_map_image(proj, rgba_neg_map_image)

            pos_thr = self.exchange_data["pos_thresholds"]
            self.pos_map_thresholds_widget.set_thresholds(pos_thr)

            if not self.exchange_data["is_rtqa"] and self.negMapCheckBox.isChecked():
                neg_thr = self.exchange_data["neg_thresholds"]
                self.neg_map_thresholds_widget.set_thresholds(neg_thr)

            if con.use_roi:
                roi_t = []
                roi_c = []
                roi_s = []
                for i in self.selected_roi:
                    roi_t.append(self.exchange_data["ROI_t"][i])
                    roi_c.append(self.exchange_data["ROI_c"][i])
                    roi_s.append(self.exchange_data["ROI_s"][i])

                self.orthView.set_roi(projview.ProjectionType.transversal, roi_t, self.selected_roi)
                self.orthView.set_roi(projview.ProjectionType.coronal, roi_c, self.selected_roi)
                self.orthView.set_roi(projview.ProjectionType.sagittal, roi_s, self.selected_roi)

            if self.orthViewInitialize:
                self.orthView.reset_view()

            self.orthViewInitialize = False

    # --------------------------------------------------------------------------
    def closeEvent(self, event):

        self.exchange_data["is_stopped"] = True
        if self._core_process.is_alive():
            self._core_process.join()
        if self._view_form_process.is_alive():
            self._view_form_process.join()

        self.close_shmem()

        self.exchange_data = None
        self.close()
        print("main process finished")

    # --------------------------------------------------------------------------
    def close_shmem(self):

        self.mc_shmem.close()
        self.mc_shmem.unlink()
        self.mosaic_shmem.close()
        self.mosaic_shmem.unlink()
        self.epi_shmem.close()
        self.epi_shmem.unlink()
        self.stat_shmem.close()
        self.stat_shmem.unlink()
        self.proj_t_shmem.close()
        self.proj_t_shmem.unlink()
        self.proj_c_shmem.close()
        self.proj_c_shmem.unlink()
        self.proj_s_shmem.close()
        self.proj_s_shmem.unlink()
        self.nfb_shmem.close()
        self.nfb_shmem.unlink()
        self.ts_shmem.close()
        self.ts_shmem.unlink()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = OpenNFTManager()
    sys.exit(app.exec_())
