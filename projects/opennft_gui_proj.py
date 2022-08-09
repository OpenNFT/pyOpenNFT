import enum
import time
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import multiprocessing as mp

from multiprocessing import shared_memory
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget
from loguru import logger

import opennft_console_proj
from opennft import mosaicview, projview, mapimagewidget, volviewformation, LegacyNftConfigLoader
from opennft import colors as col
from opennft.config import config as con


class ImageViewMode(str, enum.Enum):
    mosaic = 'mosaic'
    orthview_anat = 'orthview_anat'
    orthview_epi = 'orthview_epi'


class OpenNFTManager(QWidget):
    def __init__(self, *args, **kwargs):

        self.init_exchange_data()

        # setup button routine
        self._core_process = opennft_console_proj.OpenNFTCoreProj(self.exchange_data)

        self.config = self._core_process.config
        self.session = self._core_process.session
        self.nr_vol = self.exchange_data["nr_vol"]
        self.nr_rois = self.exchange_data["nr_rois"]
        self.vol_dim = self.exchange_data["vol_dim"]
        self.mosaic_dim = self.exchange_data["mosaic_dim"]
        self.view_form_init()

        self.init_shmem()

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
            self.cbImageViewMode.setEnabled(True)
            self.orthView.cursorPositionChanged.connect(self.onChangeOrthViewCursorPosition)

            self.mcPlot = self.createMcPlot(self.layoutPlot1)
            self.rawRoiPlot, self.procRoiPlot, self.normRoiPlot = self.createRoiPlots()
            self.createMusterInfo()
            self.btnStart.clicked.connect(self.onStart)
            self.guiTimer = QTimer(self)
            self.guiTimer.timeout.connect(self.onCheckGUIUpdated)
            self.show()
        else:
            # self.setup()
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

        time_series_array = np.zeros((3, self.nr_vol, self.nr_rois), dtype=np.float32)
        self.ts_shmem = shared_memory.SharedMemory(create=True, size=time_series_array.nbytes, name=con.shmem_file_names[8])
        self.ts_data = np.ndarray(shape=time_series_array.shape, dtype=time_series_array.dtype, buffer=self.ts_shmem.buf)

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

    def createMusterInfo(self):
        # TODO: More general way to use any protocol
        tmpCond = list()
        nrCond = list()
        for c in self.session.offsets:
            tmpCond.append(c)
            nrCond.append(tmpCond[-1].shape[0])

        if not ('BAS' in self.session.prot_names):  # implicit baseline
            # self.P['ProtCond'][0] - 0 is for Baseline indexes
            tmpCond.insert(0, np.array([[t[n] for t in self.session.prot_cond[0]] for n in (0,-1)]).T)
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
        plotitem.showGrid(x=grid, y=grid, alpha=con.plot_grid_alpha)

    def makeRoiPlotLegend(self):
        roiNames = []

        for roiName in self.exchange_data["roi_names"]:
            roiName = Path(roiName).stem
            if len(roiName) > con.max_roi_name_length:
                roiName = roiName[:2] + '..' + roiName[-2:]
            roiNames.append(roiName)

        self.labelPlotLegend.setText('')
        legendText = '<html><head/><body><p>'

        numRoi = int(self.nr_rois)

        for i, n, c in zip(range(1, numRoi + 1), roiNames, col.ROI_PLOT_COLORS):
            cname = pg.mkPen(color=c).color().name()
            legendText += (
                    '<span style="font-weight:600;color:{};">'.format(cname)
                    + 'ROI_{} {}</span>, '.format(i, n))

        legendText += (
            '<span style="font-weight:600;color:k;">Operation: {}</span>'.format(self.config.roi_anat_operation))
        legendText += '</p></body></html>'

        self.labelPlotLegend.setText(legendText)

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

    def drawMusterPlot(self, plotitem: pg.PlotItem):
        ylim = con.muster_y_limits

        # For autoRTQA mode
        if True:
            self.computeMusterPlotData(ylim)
            muster = []

            for i in range(self.musterInfo['condTotal']):
                muster.append(
                    plotitem.plot(x=self.musterInfo['xCond' + str(i + 1)],
                                  y=self.musterInfo['yCond' + str(i + 1)],
                                  fillLevel=ylim[0],
                                  pen=col.MUSTER_PEN_COLORS[i],
                                  brush=col.MUSTER_BRUSH_COLORS[i])
                )

        else:
            muster = [
                plotitem.plot(x=[1, self.nr_vol],
                              y=[-1000, 1000],
                              fillLevel=ylim[0],
                              pen=col.MUSTER_PEN_COLORS[9],
                              brush=col.MUSTER_BRUSH_COLORS[9])
            ]

        return muster

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

    def drawRoiPlots(self, init, data):

        dataRaw = np.array(data[0,:,:].squeeze().T, ndmin=2)
        dataProc = np.array(data[1,:,:].squeeze().T, ndmin=2)
        dataNorm = np.array(data[2,:,:].squeeze().T, ndmin=2)

        # TODO: plot_feedback
        # if self.config.plot_feedback:
        #     dataNorm = np.concatenate(
        #         (dataNorm, np.array([self.displaySamples]) / self.config.max_feedback_val)
        #     )

        self.drawGivenRoiPlot(init, self.rawRoiPlot, dataRaw)
        self.drawGivenRoiPlot(init, self.procRoiPlot, dataProc)
        self.drawGivenRoiPlot(init, self.normRoiPlot, dataNorm)

    def drawGivenRoiPlot(self, init, plotwidget: pg.PlotWidget, data):
        plotitem = plotwidget.getPlotItem()

        sz, l = data.shape

        if init:

            plotitem.enableAutoRange(enable=True, x=False, y=True)

            plotitem.clear()
            muster = self.drawMusterPlot(plotitem)

            plots = []

            plot_colors = np.array(col.ROI_PLOT_COLORS)
            if self.config.max_feedback_val:
                plot_colors = np.append(plot_colors, col.ROI_PLOT_COLORS[int(self.nr_rois)])
            for i, c in zip(range(sz), plot_colors):
                pen = pg.mkPen(color=c, width=con.roi_plot_width)
                p = plotitem.plot(pen=pen)
                plots.append(p)

            self.drawGivenRoiPlot.__dict__[plotitem] = plots, muster

        x = np.arange(1, l + 1, dtype=np.float64)

        for p, y in zip(self.drawGivenRoiPlot.__dict__[plotitem][0], data):
            p.setData(x=x, y=np.array(y))

        # if self.config.prot != 'InterBlock':
        #     if plotwidget == self.procRoiPlot:
        #         posMin = np.array(self.outputSamples['posMin'], ndmin=2)
        #         posMax = np.array(self.outputSamples['posMax'], ndmin=2)
        #         inds = list(self.selectedRoi)
        #         inds.append(len(posMin) - 1)
        #         posMin = posMin[inds]
        #         posMax = posMax[inds]
        #
        #         self.drawMinMaxProcRoiPlot(
        #             init, data,
        #             posMin, posMax)

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

    def onStart(self):
        if not self._core_process.is_alive():
            self.setupRoiPlots()
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

        if not self.exchange_data["is_stopped"]:

            if self.exchange_data["data_ready_flag"]:
                self.drawMcPlots(self.mcPlot, self.mc_data)

                iter_norm_number = self.exchange_data["iter_norm_number"]
                data = self.ts_data[:,:iter_norm_number,:]
                self.drawRoiPlots(True, data)

                self.exchange_data["data_ready_flag"] = False

            if self.exchange_data["view_mode"] == ImageViewMode.mosaic:

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
