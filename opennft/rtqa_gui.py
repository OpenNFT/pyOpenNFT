# -*- coding: utf-8 -*-
from PyQt6 import QtWidgets, QtCore, uic

import numpy as np
import multiprocessing as mp
import pyqtgraph as pg
import itertools

from PyQt6.QtCore import QTimer

from opennft import setting_utils
from loguru import logger
from opennft import constants as cons


class RTQAWindow(QtWidgets.QWidget):
    """Real-time quality assessment GUI class
    """

    # --------------------------------------------------------------------------
    def __init__(self, rtqa_calc, input, output):
        super(RTQAWindow, self).__init__(flags=QtCore.Qt.WindowType.Window)
        mp.Process.__init__(self)

        uic.loadUi(setting_utils.get_ui_file('rtqa.ui'), self)

        self.input = input
        self.output = output

        self.default_fd_thresholds = input["default_fd_thresholds"]
        self.default_dvars_threshold = input["default_dvars_threshold"]

        # parent data transfer block
        sz = int(input["nr_rois"])
        self.nr_rois = sz
        self.musterInfo = input["muster_info"]
        self.motion_names = ['X', 'Y', 'Z', 'Pitch', 'Roll', 'Yaw', 'FD']
        self.init = True

        if input["is_auto_rtqa"]:
            self.comboBox.model().item(2).setEnabled(False)
            self.comboBox.model().item(6).setEnabled(False)
        else:
            self.compute_muster_plot_data(cons.MUSTER_Y_LIMITS)

        xrange = rtqa_calc.xrange
        self.xrange = xrange

        # Additional GUI elements connection and initialization
        group_box_layout = self.roiGroupBox.layout()
        for i in range(sz):
            if i == sz - 1:
                name = 'Whole brain ROI'
            else:
                name = 'ROI_' + str(i + 1)
            label = QtWidgets.QLabel(name)
            label.setStyleSheet("color: " + cons.ROI_PLOT_COLORS[i].name())
            label.setVisible(False)
            group_box_layout.addWidget(label)
        self.selectedRoiLabels = self.roiGroupBox.findChildren(QtWidgets.QLabel)
        self.mcrRadioButton.toggled.connect(self.on_radio_button_state_changed)

        # Plots initialization
        self.snrPlot = pg.PlotWidget(self)
        self.snrPlot.setBackground((255, 255, 255))
        self.snrPlotLayout.addWidget(self.snrPlot)
        p = self.snrPlot.getPlotItem()
        self.plots_setup(p, "SNR [a.u.]", xrange)
        self.draw_muster_plot(p)
        p.setYRange(-1, 1, padding=0.0)

        self.noRegSnrPlot = pg.PlotWidget(self)
        self.noRegSnrPlot.setBackground((255, 255, 255))
        self.noRegSnrPlotLayout.addWidget(self.noRegSnrPlot)
        p = self.noRegSnrPlot.getPlotItem()
        self.plots_setup(p, "SNR [a.u.]", xrange)
        self.draw_muster_plot(p)
        p.setYRange(-1, 1, padding=0.0)

        self.msePlot = pg.PlotWidget(self)
        self.msePlot.setBackground((255, 255, 255))
        self.msePlotLayout.addWidget(self.msePlot)
        p = self.msePlot.getPlotItem()
        self.plots_setup(p, "Mean squared error [a.u.]", xrange)
        self.draw_muster_plot(p)
        p.setYRange(-1, 1, padding=0.0)

        self.trendPlot = pg.PlotWidget(self)
        self.trendPlot.setBackground((255, 255, 255))
        self.linearTreandPlotLayout.addWidget(self.trendPlot)
        p = self.trendPlot.getPlotItem()
        self.plots_setup(p, "Beta regressor amplitude [a.u.]", xrange)
        self.draw_muster_plot(p)
        p.setYRange(-1, 1, padding=0.0)

        self.fdPlot = pg.PlotWidget(self)
        self.fdPlot.setBackground((255, 255, 255))
        self.fdPlotLayout.addWidget(self.fdPlot)
        p = self.fdPlot.getPlotItem()
        self.plots_setup(p, "FD [mm]", xrange)

        self.translatPlot = pg.PlotWidget(self)
        self.translatPlot.setBackground((255, 255, 255))
        self.tdPlotLayout.addWidget(self.translatPlot)
        p = self.translatPlot.getPlotItem()
        self.plots_setup(p, "Amplitude [mm]", xrange)

        self.rotatPlot = pg.PlotWidget(self)
        self.rotatPlot.setBackground((255, 255, 255))
        self.rdPlotLayout.addWidget(self.rotatPlot)
        p = self.rotatPlot.getPlotItem()
        self.plots_setup(p, "Amplitude [mm]", xrange)

        self.dvarsPlot = pg.PlotWidget(self)
        self.dvarsPlot.setBackground((255, 255, 255))
        self.dvarsPlotLayout.addWidget(self.dvarsPlot)
        p = self.dvarsPlot.getPlotItem()
        self.plots_setup(p, "DVARS [a.u.]", xrange)

        self.spikesPlot = pg.PlotWidget(self)
        self.spikesPlot.setBackground((255, 255, 255))
        self.spikesPlotLayout.addWidget(self.spikesPlot)
        p = self.spikesPlot.getPlotItem()
        self.plots_setup(p, "Amplitude [a.u.]", xrange)
        self.draw_muster_plot(p)
        p.setYRange(-1, 1, padding=0.0)

        # CNR, means and variances plots and labels
        if not input["is_auto_rtqa"]:

            self.cnrPlot = pg.PlotWidget(self)
            self.cnrPlot.setBackground((255, 255, 255))
            self.cnrPlotLayout.addWidget(self.cnrPlot)
            p = self.cnrPlot.getPlotItem()
            self.plots_setup(p, "CNR [a.u.]", xrange)
            self.draw_muster_plot(p)
            p.setYRange(-1, 1, padding=0.0)

            self.meanPlot = pg.PlotWidget(self)
            self.meanPlot.setBackground((255, 255, 255))
            self.meanPlotLayout.addWidget(self.meanPlot)
            p = self.meanPlot.getPlotItem()
            self.plots_setup(p, "Mean [a.u.]", xrange)
            self.draw_muster_plot(p)
            p.setYRange(-1, 1, padding=0.0)

            self.varPlot = pg.PlotWidget(self)
            self.varPlot.setBackground((255, 255, 255))
            self.varPlotLayout.addWidget(self.varPlot)
            p = self.varPlot.getPlotItem()
            self.plots_setup(p, "Variance [a.u.]", xrange)
            self.draw_muster_plot(p)
            p.setYRange(-1, 1, padding=0.0)

            names = ['ROI_1 rMean', ' bas', ' cond']
            color = [cons.ROI_PLOT_COLORS[0], cons.ROI_PLOT_COLORS[0], cons.ROI_PLOT_COLORS[0]]
            for i in range(sz - 1):
                if i == sz - 2:
                    name = 'Whole brain ROI'
                else:
                    name = 'ROI_' + str(i + 2)
                names.append(name + ' rMean')
                names.append(' bas')
                names.append(' cond')
                color = color + [cons.ROI_PLOT_COLORS[i + 1]] + [cons.ROI_PLOT_COLORS[i + 1]] + [
                    cons.ROI_PLOT_COLORS[i + 1]]
            pens = []
            for i in range(sz * 3):
                pens = pens + [pg.mkPen(color[i], width=1.2)]
            self.make_text_value_label(self.labelMean, names, pens)

            names = ['ROI_1 rVariance', ' bas', ' cond']
            for i in range(sz - 1):
                if i == sz - 2:
                    name = 'Whole brain ROI'
                else:
                    name = 'ROI_' + str(i + 2)
                names.append(name + ' rVariance')
                names.append(' bas')
                names.append(' cond')
            self.make_text_value_label(self.labelVar, names, pens)

        # Other labels initialization
        names = ['Translationals: ']
        pens = [cons.PLOT_PEN_COLORS[6]]
        for i in range(3):
            names.append(self.motion_names[i])
            pens.append(cons.PLOT_PEN_COLORS[i])
        self.make_text_value_label(self.tdLabel, names, pens)

        names = ['Rotations: ']
        pens = [cons.PLOT_PEN_COLORS[6]]
        for i in range(3):
            names.append(self.motion_names[i + 3])
            pens.append(cons.PLOT_PEN_COLORS[i + 3])
        self.make_text_value_label(self.rdLabel, names, pens)

        names = ['Framewise Displacement']
        pens = [cons.PLOT_PEN_COLORS[0]]
        for i in range(len(self.default_fd_thresholds) - 1):
            names.append('Threshold ' + str(i + 1))
            pens.append(cons.PLOT_PEN_COLORS[i + 1])
        self.make_text_value_label(self.fdLabel, names, pens)

        self.threshold = self.default_fd_thresholds
        self.roi_checked()
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)

        self.rtqa_draw_timer = QTimer(self)
        self.rtqa_draw_timer.timeout.connect(self.draw_rtqa)

    # --------------------------------------------------------------------------
    def closeEvent(self, event):

        self.hide()
        event.accept()

    # --------------------------------------------------------------------------
    def plots_setup(self, p, yName, xrange):

        p.setLabel('left', yName)
        p.setMenuEnabled(enableMenu=False)
        p.setMouseEnabled(x=False, y=False)
        p.showGrid(x=True, y=True, alpha=1)
        p.installEventFilter(self)
        p.disableAutoRange(axis=pg.ViewBox.XAxis)
        p.setXRange(1, xrange, padding=0.0)

    # --------------------------------------------------------------------------
    def rtqa_vol_state(self):

        state = self.comboBox.currentIndex()
        self.output["show_vol"] = self.volumeCheckBox.isChecked()
        if state == 0:
            self.input["which_vol"] = 0
        elif state == 2:
            self.input["which_vol"] = 2

    # --------------------------------------------------------------------------
    def on_combobox_changed(self):
        """  SNR/CNR label switching. Both modes use the same label
        """

        state = self.comboBox.currentIndex()

        # SNR state
        if state == 0:

            names = ['SNR ']
            pens = [cons.PLOT_PEN_COLORS[6]]
            sz = self.nr_rois
            for i in range(sz):
                if i == sz - 1:
                    name = 'Whole brain ROI'
                else:
                    name = 'ROI_' + str(i + 1)
                names.append(name + ':  ' + '{0:.3f}'.format(float(self.output["rSNR"][i][self.input["iteration"]])))
                pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))
            self.make_text_value_label(self.valuesLabel, names, pens, lineBreak='<br>')
            self.input["which_vol"] = 0

        # CNR state
        elif state == 2:

            self.stackedWidgetOptions.setCurrentIndex(0)

            names = ['СNR ']
            pens = [cons.PLOT_PEN_COLORS[6]]
            sz = self.nr_rois
            for i in range(sz):
                if i == sz - 1:
                    name = 'Whole brain ROI'
                else:
                    name = 'ROI_' + str(i + 1)
                names.append(name + ':  ' + '{0:.3f}'.format(float(self.output["rCNR"][i][self.input["iteration"]])))
                pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))
            self.make_text_value_label(self.valuesLabel, names, pens, lineBreak='<br>')
            self.input["which_vol"] = 2

        if state == 1 or state == 7:
            self.roiGroupBox.setVisible(False)
        else:
            self.roiGroupBox.setVisible(True)

        if self.input["is_stopped"] and self.input["iteration"] != 1:
            self.draw_rtqa()

    # --------------------------------------------------------------------------
    def on_radio_button_state_changed(self):
        """ FD and MD mode change. Mode changing switch plots and plot title
        """

        if self.mcrRadioButton.isChecked():
            names = ['Micro Displacement']
            pens = [cons.PLOT_PEN_COLORS[0]]
            names.append('Threshold')
            pens.append(cons.PLOT_PEN_COLORS[2])
            self.make_text_value_label(self.fdLabel, names, pens)

        else:
            names = ['Framewise Displacement']
            pens = [cons.PLOT_PEN_COLORS[0]]
            for i in range(len(self.default_fd_thresholds) - 1):
                names.append('Threshold ' + str(i + 1))
                pens.append(cons.PLOT_PEN_COLORS[i + 1])
            self.make_text_value_label(self.fdLabel, names, pens)

        self.draw_mc_plots(self.mcrRadioButton.isChecked())

    # --------------------------------------------------------------------------
    def make_text_value_label(self, label, names, pens, lineBreak=' '):
        """ Dynamic generation of titles and value labels

        :param label: label for text update
        :param names: set of names
        :param pens: set of pens for each name
        :param lineBreak: line break for value labels, space by default for title labels
        """

        label.setText('')
        legendText = '<html><head/><body><p>'

        for n, c in zip(names, pens):
            cname = c.color().name()
            legendText += (
                    '<span style="font-weight:600;color:{};">'.format(cname) + '{}</span>'.format(n) + lineBreak)

        legendText += '</p></body></html>'

        label.setText(legendText)

    # --------------------------------------------------------------------------
    def roi_checked(self):
        """ Redrawing plots when the set of selected ROIs is changed even if run is stopped
        """

        for i in range(len(self.selectedRoiLabels)):
            if i in self.input["roi_checked"]:
                self.selectedRoiLabels[i].setVisible(True)
            else:
                self.selectedRoiLabels[i].setVisible(False)

        self.init = True

    # --------------------------------------------------------------------------
    def compute_muster_plot_data(self, ylim):
        singleY = np.array([ylim[0], ylim[1], ylim[1], ylim[0]])

        def compute_conds(nrCond, tmpCond):
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
            xCond, yCond = compute_conds(self.musterInfo['nrCond' + str(cond + 1)],
                                        self.musterInfo['tmpCond' + str(cond + 1)])
            self.musterInfo['xCond' + str(cond + 1)] = xCond
            self.musterInfo['yCond' + str(cond + 1)] = yCond

    # --------------------------------------------------------------------------
    def draw_muster_plot(self, plotitem):

        ylim = cons.MUSTER_Y_LIMITS

        if self.comboBox.model().item(2).isEnabled():

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
                plotitem.plot(x=[1, self.xrange],
                              y=[-1000, 1000],
                              fillLevel=ylim[0],
                              pen=cons.MUSTER_PEN_COLORS[9],
                              brush=cons.MUSTER_BRUSH_COLORS[9])
            ]

        return muster

    # --------------------------------------------------------------------------
    def draw_ts(self, init, plotitem, data, checked_boxes_ind):
        """ Time-series plot method

        :param init: flag for plot initializtion
        :param plotitem: time-series plotitem
        :param data: time-series value for drawing
        :param checked_boxes_ind: indexes of selected ROIs
        """

        if self.tsCheckBox.isChecked():

            sz, l = data.shape

            if init:

                plotitem.clear()
                plots = []

                muster = self.draw_muster_plot(plotitem)

                plot_colors = np.array(cons.ROI_PLOT_COLORS)[checked_boxes_ind]
                for i, c in zip(range(sz), plot_colors):
                    pen = pg.mkPen(color=c, width=cons.ROI_PLOT_WIDTH)
                    p = plotitem.plot(pen=pen)
                    plots.append(p)

                self.draw_ts.__dict__[plotitem] = plots, muster

            x = np.arange(1, l + 1, dtype=np.float64)

            plotitems = self.draw_ts.__dict__[plotitem][0]
            for p, y in zip(plotitems, data):
                p.setData(x=x, y=np.array(y))

            items = plotitem.listDataItems()

            for m in self.draw_ts.__dict__[plotitem][1]:
                items.remove(m)

            if data.any():
                if plotitem.vb.state["targetRange"][1] == [-1, 1]:
                    plotitem.enableAutoRange(enable=True, x=False, y=True)
                plotitem.setYRange(np.min(data).astype(np.float32), np.max(data).astype(np.float32), padding=0.0)

    # --------------------------------------------------------------------------
    def draw_rtqa(self):
        """ Encapsulated plots drawing

        :param n: last volume index
        """

        if self.isVisible():

            if self.input["calc_ready"]:

                n = self.input["iteration"]
                self.roi_checked()
                checked_boxes_ind = self.input["roi_checked"]
                sz = self.nr_rois
                index_volume = n - 1

                current_menu = self.comboBox.currentIndex()

                if current_menu == 0:

                    # SNR plot
                    plotitem = self.snrPlot.getPlotItem()
                    data = self.output["rSNR"][checked_boxes_ind, 0:n]
                    self.draw_ts(self.init, plotitem, data, checked_boxes_ind)

                    names = ['SNR ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    for i in range(sz):
                        if i == sz - 1:
                            name = 'Whole brain ROI'
                        else:
                            name = 'ROI_' + str(i + 1)
                        names.append(name + ': ' + '{0:.3f}'.format(float(self.output["rSNR"][i, index_volume])))
                        pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))

                    self.make_text_value_label(self.valuesLabel, names, pens, lineBreak='<br>')

                elif current_menu == 1:

                    self.draw_mc_plots(self.mcrRadioButton.isChecked())

                    names = ['<u>FD</u> ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    names.append('Threshold 1: ' + str(int(self.output["excFD"][0])))
                    pens.append(cons.PLOT_PEN_COLORS[1])
                    names.append('Threshold 2: ' + str(int(self.output["excFD"][1])))
                    pens.append(cons.PLOT_PEN_COLORS[2])
                    names.append('<br><u>MD</u> ')
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    names.append('Threshold: ' + str(int(self.output["excMD"])))
                    pens.append(cons.PLOT_PEN_COLORS[2])
                    names.append('<br><u>Mean FD</u> ')
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    names.append('{0:.3f}'.format(self.output["meanFD"]))
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    names.append('<br><u>Mean MD</u> ')
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    names.append('{0:.3f}'.format(self.output["meanMD"]))
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    names.append('<br><u>Offset MC parameters</u> ')
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    for i in range(6):
                        names.append('{0:.3e}'.format(self.input["offset_mc"][i]))
                        pens.append(cons.PLOT_PEN_COLORS[6])
                    self.make_text_value_label(self.mcmdValuesLabel, names, pens, lineBreak='<br>')

                elif current_menu == 2:

                    plotitem = self.cnrPlot.getPlotItem()
                    data = self.output["rCNR"][checked_boxes_ind, 0:n]
                    self.draw_ts(self.init, plotitem, data, checked_boxes_ind)

                    # Means plot
                    plotitem = self.meanPlot.getPlotItem()
                    data = np.append(self.output["rMean"][checked_boxes_ind, 0:n],
                                     self.output["meanBas"][checked_boxes_ind, 0:n], axis=0)
                    data = np.append(data, self.output["meanCond"][checked_boxes_ind, 0:n], axis=0)
                    color = np.array(cons.ROI_PLOT_COLORS)[checked_boxes_ind]
                    color = np.append(color, np.array(cons.ROI_PLOT_COLORS)[checked_boxes_ind])
                    color = np.append(color, np.array(cons.ROI_PLOT_COLORS)[checked_boxes_ind])
                    style = [QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenStyle.DashLine,
                             QtCore.Qt.PenStyle.DashDotDotLine]
                    self.draw_stat_values(self.init, plotitem, data, color, style)

                    # Variances plot
                    plotitem = self.varPlot.getPlotItem()
                    data = np.append(self.output["rVar"][checked_boxes_ind, 0:n],
                                     self.output["varBas"][checked_boxes_ind, 0:n], axis=0)
                    data = np.append(data, self.output["varCond"][checked_boxes_ind, 0:n], axis=0)
                    self.draw_stat_values(self.init, plotitem, data, color, style)

                    names = ['СNR ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    for i in range(self.nr_rois):
                        if i == sz - 1:
                            name = 'Whole brain ROI'
                        else:
                            name = 'ROI_' + str(i + 1)
                        names.append(name + ': ' + '{0:.3f}'.format(float(self.output["rCNR"][i][index_volume - 1])))
                        pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))

                    if self.comboBox.currentIndex() == 2:
                        names.append('<br><br>Baseline values   --- ')
                        pens.append(cons.PLOT_PEN_COLORS[6])
                        names.append('Condition values -··-··- ')
                        pens.append(cons.PLOT_PEN_COLORS[6])

                    self.make_text_value_label(self.valuesLabel, names, pens, lineBreak='<br>')

                elif current_menu == 3:

                    # Spikes plot
                    plotitem = self.spikesPlot.getPlotItem()
                    data = self.output["glmProcTimeSeries"][checked_boxes_ind, 0:n]
                    self.draw_spikes(self.init, plotitem, data, checked_boxes_ind)

                    # Spikes labels
                    cnt = 0
                    for i in range(sz):
                        cnt = cnt + np.count_nonzero(self.output["posSpikes"][str(i)])
                    names = ['( Circles ) <br>Positive spikes: ' + str(int(cnt))]

                    cnt = 0
                    for i in range(sz):
                        cnt = cnt + np.count_nonzero(self.output["negSpikes"][str(i)])
                    names.append('<br>( Diamonds )<br>Negative spikes: ' + str(int(cnt)))
                    pens = [cons.PLOT_PEN_COLORS[6],
                            cons.PLOT_PEN_COLORS[6]]
                    self.make_text_value_label(self.spikesLabel, names, pens, lineBreak='<br>')

                elif current_menu == 4:

                    # Kalman filter MSE plot
                    plotitem = self.msePlot.getPlotItem()
                    data = self.output["rMSE"][checked_boxes_ind, 0:n]
                    self.draw_ts(self.init, plotitem, data, checked_boxes_ind)

                    # MSE label
                    names = ['MSE ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    for i in range(sz):
                        if i == sz - 1:
                            name = 'Whole brain ROI'
                        else:
                            name = 'ROI_' + str(i + 1)
                        names.append(name + ': ' + '{0:.3f}'.format(float(self.output["rMSE"][i, index_volume])))
                        pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))

                    self.make_text_value_label(self.mseLabel, names, pens, lineBreak='<br>')

                elif current_menu == 5:

                    # Linear trend coefficients plot
                    plotitem = self.trendPlot.getPlotItem()
                    data = self.output["linTrendCoeff"][checked_boxes_ind, 0:n]
                    self.draw_ts(self.init, plotitem, data, checked_boxes_ind)

                    # Linear trend labels
                    names = ['Linear trend beta ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    for i in range(sz):
                        if i == sz - 1:
                            name = 'Whole brain ROI'
                        else:
                            name = 'ROI_' + str(i + 1)
                        names.append(
                            name + ': ' + '{0:.3f}'.format(float(self.output["linTrendCoeff"][i, index_volume - 1])))
                        pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))
                    self.make_text_value_label(self.trendLabel, names, pens, lineBreak='<br>')

                elif current_menu == 6:

                    # No regulation SNR plot
                    plotitem = self.noRegSnrPlot.getPlotItem()
                    data = self.output["rNoRegSNR"][checked_boxes_ind, 0:n]
                    self.draw_ts(self.init, plotitem, data, checked_boxes_ind)

                    # No regulation SNR label
                    names = ['no reg SNR ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    for i in range(sz):
                        if i == sz - 1:
                            name = 'Whole brain ROI'
                        else:
                            name = 'ROI_' + str(i + 1)
                        names.append(name + ': ' + '{0:.3f}'.format(float(self.output["rNoRegSNR"][i, index_volume])))
                        pens.append(pg.mkPen(color=cons.ROI_PLOT_COLORS[i], width=1.2))

                    self.make_text_value_label(self.noRegSnrValueLabel, names, pens, lineBreak='<br>')

                elif current_menu == 7:

                    # DVARS plot
                    plotitem = self.dvarsPlot.getPlotItem()
                    plotitem.clear()
                    plotitem.plot(y=self.output["DVARS"], pen=cons.PLOT_PEN_COLORS[0], name='DVARS')
                    plotitem.plot(x=np.arange(1, self.xrange + 1, dtype=np.float64),
                                  y=self.default_dvars_threshold * np.ones(self.xrange),
                                  pen=cons.PLOT_PEN_COLORS[2], name='thr')

                    # DVARS label
                    names = ['DVARS ']
                    pens = [cons.PLOT_PEN_COLORS[6]]
                    names.append('{0:.3f} '.format(float(self.output["DVARS"][-1])))
                    pens.append(cons.PLOT_PEN_COLORS[6])
                    names.append('<br>Threshold : ' + str(int(self.output["excDVARS"])))
                    pens.append(cons.PLOT_PEN_COLORS[6])

                    self.make_text_value_label(self.dvarsLabel, names, pens, lineBreak='<br>')

                self.init = False

            if not self.input["is_stopped"]:
                self.input["calc_ready"] = False

    # --------------------------------------------------------------------------
    def draw_stat_values(self, init, plotitem, data, color, style):
        """ Drawing method for mean and variance statistics

        :param init: flag for plot initializtion
        :param plotitem: mean or variance plotitem
        :param data: signal values for drawing
        :param color: color of each ROI line
        :param style: style of each ROI line
        """

        if self.tsCheckBox.isChecked():

            sz, l = data.shape

            if init:

                plotitem.clear()
                plots = []

                muster = self.draw_muster_plot(plotitem)

                style = np.repeat(style, sz / 3)

                for i, c, s in zip(range(sz), color, style):
                    pen = pg.mkPen(c, width=3.0, style=QtCore.Qt.PenStyle(s))
                    p = plotitem.plot(pen=pen)
                    plots.append(p)

                self.draw_ts.__dict__[plotitem] = plots, muster

            x = np.arange(1, l + 1, dtype=np.float64)

            for p, y in zip(self.draw_ts.__dict__[plotitem][0], data):
                p.setData(x=x, y=np.array(y))

            items = plotitem.listDataItems()

            for m in self.draw_ts.__dict__[plotitem][1]:
                items.remove(m)

            if data.any():
                plotitem.setYRange(np.min(data[np.nonzero(data)]), np.max(data), padding=0.0)

    # --------------------------------------------------------------------------
    def draw_spikes(self, init, plotitem, data, checked_boxes_ind):
        """ Spikes plot drawing

        :param init: flag for plot initializtion
        :param plotitem: spikes plotitem
        :param data: signal values for drawing
        :param checked_boxes_ind: indexes of selected ROIs
        """

        # First part - line drawing
        sz, l = data.shape
        x = np.arange(1, l + 1, dtype=np.float64)

        if init:
            plotitem.clear()
            plots = []

            muster = self.draw_muster_plot(plotitem)

            for i, c in zip(range(sz), np.array(cons.ROI_PLOT_COLORS)[checked_boxes_ind]):
                pen = pg.mkPen(color=c, width=cons.ROI_PLOT_WIDTH)
                p = plotitem.plot(pen=pen)
                plots.append(p)

            self.draw_spikes.__dict__[plotitem] = plots, muster

        plots = self.draw_spikes.__dict__[plotitem][0]
        for p, y in zip(plots, data):
            p.setData(x=x, y=np.array(y))

        # Second part - spikes marking
        for i, c in zip(range(sz), np.array(cons.ROI_PLOT_COLORS)[checked_boxes_ind]):

            roiInd = checked_boxes_ind[i]
            if self.output["posSpikes"][str(roiInd)].any():
                brush = pg.mkBrush(color=c)
                p = plotitem.scatterPlot(symbol='o', size=20, brush=brush)
                plots.append(p)
                plots[-1].setData(x=self.output["posSpikes"][str(roiInd)] + 1,
                                  y=self.output["glmProcTimeSeries"][roiInd, self.output["posSpikes"][str(roiInd)]])

                pen = pg.mkPen(color=pg.mkColor(0, 0, 0), width=1.5 * cons.ROI_PLOT_WIDTH)
                p = plotitem.plot(pen=pen)
                plots.append(p)

                inds = self.output["posSpikes"][str(roiInd)]
                indX = np.array(list(itertools.chain.from_iterable(zip(inds, inds + 1))))
                indY = np.array(list(itertools.chain.from_iterable(zip(inds - 1, inds))))

                y = np.array(self.output["glmProcTimeSeries"][roiInd, indY])
                x1 = indX

                plots[-1].setData(x=x1, y=y, connect='pairs')

            if self.output["negSpikes"][str(roiInd)].any():
                brush = pg.mkBrush(color=c)
                p = plotitem.scatterPlot(symbol='d', size=20, brush=brush)
                plots.append(p)
                plots[-1].setData(x=self.output["negSpikes"][str(roiInd)] + 1,
                                  y=self.output["glmProcTimeSeries"][roiInd, self.output["negSpikes"][str(roiInd)]])

                pen = pg.mkPen(color=pg.mkColor(0, 0, 0), width=1.5 * cons.ROI_PLOT_WIDTH)
                p = plotitem.plot(pen=pen)
                plots.append(p)

                inds = self.output["negSpikes"][str(roiInd)]
                indX = np.array(list(itertools.chain.from_iterable(zip(inds, inds + 1))))
                indY = np.array(list(itertools.chain.from_iterable(zip(inds - 1, inds))))

                y = np.array(self.output["glmProcTimeSeries"][roiInd, indY])
                x1 = indX

                plots[-1].setData(x=x1, y=y, connect='pairs')

        items = plotitem.listDataItems()

        for m in self.draw_spikes.__dict__[plotitem][1]:
            items.remove(m)

        if data.any():
            plotitem.setYRange(np.min(self.output["glmProcTimeSeries"]) - 1,
                               np.max(self.output["glmProcTimeSeries"]) + 1, padding=0.0)

    # --------------------------------------------------------------------------
    def draw_mc_plots(self, mdFlag):

        if self.output["mc_params"].any():
            x = np.arange(1, self.output["mc_params"].shape[0] + 1, dtype=np.float64)

            self.translatPlot.clear()
            self.rotatPlot.clear()
            self.fdPlot.clear()

            for i in range(0, 3):
                self.translatPlot.plot(x=x, y=self.output["mc_params"][:, i], pen=cons.PLOT_PEN_COLORS[i],
                                       name=self.motion_names[i])

            for i in range(3, 6):
                self.rotatPlot.plot(x=x, y=self.output["mc_params"][:, i] * 50, pen=cons.PLOT_PEN_COLORS[i],
                                    name=self.motion_names[i])

            x = np.arange(1, self.output["FD"].shape[0] + 1, dtype=np.float64)

            if mdFlag:
                self.fdPlot.setLabel('left', "MD [mm]")
                self.fdPlot.plot(x=x, y=self.output["MD"], pen=cons.PLOT_PEN_COLORS[0], name='MD')
                self.fdPlot.plot(x=np.arange(0, self.xrange, dtype=np.float64),
                                 y=self.threshold[0] * np.ones(self.xrange),
                                 pen=cons.PLOT_PEN_COLORS[2], name='thr')
            else:
                self.fdPlot.setLabel('left', "FD [mm]")
                self.fdPlot.plot(x=x, y=self.output["FD"], pen=cons.PLOT_PEN_COLORS[0], name='FD')
                thresholds = self.threshold[1:3]
                for i, t in enumerate(thresholds):
                    self.fdPlot.plot(x=np.arange(0, self.xrange, dtype=np.float64), y=float(t) * np.ones(self.xrange),
                                     pen=cons.PLOT_PEN_COLORS[i + 1], name='thr' + str(i))
