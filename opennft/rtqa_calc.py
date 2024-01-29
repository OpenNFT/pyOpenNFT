# -*- coding: utf-8 -*-
import time

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

from scipy.io import savemat

from loguru import logger


class RTQACalculation(mp.Process):
    """Real-time quality assessment methods class
    """

    # --------------------------------------------------------------------------
    def __init__(self, input, output, exchange_data):

        mp.Process.__init__(self)
        self.input = input
        self.output = output
        self.exchange_data = exchange_data

        self.shmem_file_names = input["shmem_file_names"]

        # parent data transfer block
        sz = int(input["nr_rois"])
        self.nr_rois = sz
        self.first_snr_vol = input["first_snr_volume"]

        if input["is_auto_rtqa"]:
            self.ind_bas = 0
            self.ind_cond = 0
        else:
            self.ind_bas = np.array([])
            for interval in input["ind_bas"]:
                self.ind_bas = np.append(self.ind_bas, np.arange(interval[0] - 1, interval[1]))
            self.ind_cond = np.array([])
            for interval in input["ind_cond"]:
                self.ind_cond = np.append(self.ind_cond, np.arange(interval[0] - 1, interval[1]))
            self.ind_cond = np.array(self.ind_cond)

        xrange = int(input["xrange"])
        self.xrange = xrange

        # main class data initialization block
        self.fd = np.array([])
        self.mean_fd = 0
        self.md = np.array([])
        self.mean_md = 0
        self.exc_fd = [0, 0]
        self.exc_md = 0
        self.exc_fd_indexes_1 = np.array([-1])
        self.exc_fd_indexes_2 = np.array([-1])
        self.exc_md_indexes = np.array([-1])
        self.rsq_displ = np.array([0])
        self.mc_params = np.array([[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05]])
        self.radius = input["default_fd_radius"]
        self.threshold = input["default_fd_thresholds"]
        self.default_dvars_threshold = input["default_dvars_threshold"]
        self.iter_bas = 0
        self.iter_cond = 0
        self.iteration = 0
        self.block_iter = 0
        self.no_reg_block_iter = 0
        self.r_mean = np.zeros((sz, xrange))
        self.m2 = np.zeros((sz, xrange))
        self.r_var = np.zeros((sz, xrange))
        self.r_snr = np.zeros((sz, xrange))
        self.r_no_reg_mean = np.zeros((sz, xrange))
        self.no_reg_m2 = np.zeros((sz, 1))
        self.r_no_reg_var = np.zeros((sz, xrange))
        self.r_no_reg_SNR = np.zeros((sz, xrange))
        self.mean_bas = np.zeros((sz, xrange))
        self.var_bas = np.zeros((sz, xrange))
        self.m2_bas = np.zeros((sz, 1))
        self.mean_cond = np.zeros((sz, xrange))
        self.var_cond = np.zeros((sz, xrange))
        self.m2_cond = np.zeros((sz, 1))
        self.r_cnr = np.zeros((sz, xrange))
        self.glm_proc_time_series = np.zeros((sz, xrange))
        self.pos_spikes = dict.fromkeys(['{:d}'.format(x) for x in range(sz)], np.array(0))
        self.neg_spikes = dict.fromkeys(['{:d}'.format(x) for x in range(sz)], np.array(0))
        self.r_mse = np.zeros((sz, xrange))
        self.dvars = np.zeros((1,))
        self.exc_dvars = 0
        self.lin_trend_coeff = np.zeros((sz, xrange))
        self.prev_vol = np.array([])

        self.volume_data = {"mean_vol": [],
                            "m2_vol": [],
                            "var_vol": [],
                            "iter": 0,
                            "mean_bas_vol": [],
                            "m2_bas_vol": [],
                            "var_bas_vol": [],
                            "iter_bas": 0,
                            "mean_cond_vol": [],
                            "m2_cond_vol": [],
                            "var_cond_vol": [],
                            "iter_cond": 0
                            }

        self.output["rSNR"] = self.r_snr
        self.output["rCNR"] = self.r_cnr
        self.output["rMean"] = self.r_mean
        self.output["meanBas"] = self.mean_bas
        self.output["meanCond"] = self.mean_cond
        self.output["rVar"] = self.r_var
        self.output["varBas"] = self.var_bas
        self.output["varCond"] = self.var_cond
        self.output["glmProcTimeSeries"] = self.glm_proc_time_series
        self.output["rMSE"] = self.r_mse
        self.output["linTrendCoeff"] = self.lin_trend_coeff
        self.output["rNoRegSNR"] = self.r_no_reg_SNR
        self.output["DVARS"] = self.dvars
        self.output["excDVARS"] = self.exc_dvars
        self.output["mc_params"] = self.mc_params
        self.output["mc_offset"] = np.zeros((6, 1))
        self.output["FD"] = self.fd
        self.output["MD"] = self.md
        self.output["meanFD"] = self.mean_fd
        self.output["meanMD"] = self.mean_md
        self.output["excFD"] = self.exc_fd
        self.output["excMD"] = self.exc_md
        self.output["posSpikes"] = self.pos_spikes
        self.output["negSpikes"] = self.neg_spikes

    def init_shmem(self):

        nr_vol = self.exchange_data["nr_vol"]
        nr_rois = self.exchange_data["nr_rois"]

        self.mc_shmem = shared_memory.SharedMemory(name=self.shmem_file_names[0])
        self.mc_data = np.ndarray(shape=(nr_vol, 6), dtype=np.float32, buffer=self.mc_shmem.buf)

        self.ts_shmem = shared_memory.SharedMemory(name=self.shmem_file_names[1])
        self.ts_data = np.ndarray(shape=(7, nr_vol, nr_rois), dtype=np.float32, buffer=self.ts_shmem.buf)

        self.epi_shmem = shared_memory.SharedMemory(name=self.shmem_file_names[2])
        self.epi_volume = np.ndarray(shape=self.exchange_data["vol_dim"], dtype=np.float32, buffer=self.epi_shmem.buf,
                                     order="F")

        rtqa_vol_dim = tuple(self.input["dim"]) + (2,)
        self.rtqa_vol_shmem = shared_memory.SharedMemory(name=self.shmem_file_names[3])
        self.rtqa_volume = np.ndarray(shape=rtqa_vol_dim, dtype=np.float32, buffer=self.rtqa_vol_shmem.buf, order='F')

    # --------------------------------------------------------------------------
    def run(self):

        np.seterr(divide='ignore', invalid='ignore')

        self.init_shmem()

        while not self.input["is_stopped"]:

            if self.input["data_ready"]:
                self.calculate_rtqa()

                self.output["rSNR"] = self.r_snr
                self.output["rCNR"] = self.r_cnr
                self.output["rMean"] = self.r_mean
                self.output["meanBas"] = self.mean_bas
                self.output["meanCond"] = self.mean_cond
                self.output["rVar"] = self.r_var
                self.output["varBas"] = self.var_bas
                self.output["varCond"] = self.var_cond
                self.output["glmProcTimeSeries"] = self.glm_proc_time_series
                self.output["rMSE"] = self.r_mse
                self.output["linTrendCoeff"] = self.lin_trend_coeff
                self.output["rNoRegSNR"] = self.r_no_reg_SNR
                self.output["DVARS"] = self.dvars
                self.output["excDVARS"] = self.exc_dvars
                self.output["mc_params"] = self.mc_params
                self.output["FD"] = self.fd
                self.output["MD"] = self.md
                self.output["meanFD"] = self.mean_fd
                self.output["meanMD"] = self.mean_md
                self.output["excFD"] = self.exc_fd
                self.output["excMD"] = self.exc_md
                self.output["posSpikes"] = self.pos_spikes
                self.output["negSpikes"] = self.neg_spikes

                self.input["data_ready"] = False
                self.input["calc_ready"] = True

        self.rtqa_data_save()
        self.input["calc_ready"] = True

        self.mc_shmem.close()
        self.epi_shmem.close()
        self.ts_shmem.close()
        self.rtqa_vol_shmem.close()

    # --------------------------------------------------------------------------
    def calculate_rtqa(self):

        iteration = self.input["iteration"]
        for i in range(self.nr_rois):
            self.lin_trend_coeff[i][iteration] = self.input["beta_coeff"][i]

        volume = self.epi_volume

        if iteration == 0:
            self.block_iter = 0
            self.iter_bas = 0
            self.iter_cond = 0
            self.volume_data["iter"] = 0
            self.volume_data["iter_bas"] = 0
            self.volume_data["iter_cond"] = 0
            self.volume_data["mean_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["m2_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["var_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["mean_bas_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["m2_bas_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["var_bas_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["mean_cond_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["m2_cond_vol"] = np.zeros(volume.shape, order="F")
            self.volume_data["var_cond_vol"] = np.zeros(volume.shape, order="F")

        if iteration > self.first_snr_vol:
            self.calculate_rtqa_volume(volume, iteration)
        self.calculate_rtqa_ts(iteration)
        self.calculate_dvars(volume[self.input["wb_roi_indexes"][0],
                                    self.input["wb_roi_indexes"][1],
                                    self.input["wb_roi_indexes"][2]
                                   ], iteration)
        self.calc_mc()

    # --------------------------------------------------------------------------
    def calculate_rtqa_volume(self, volume, index_volume):

        tmp_output, self.volume_data["mean_vol"], \
            self.volume_data["m2_vol"], self.volume_data["var_vol"] = self.snr(self.volume_data["mean_vol"],
                                                                               self.volume_data["m2_vol"],
                                                                               volume, self.volume_data["iter"])
        self.volume_data["iter"] += 1
        output_vol = np.zeros(tmp_output.shape)
        output_vol[self.input["wb_roi_indexes"][0],
        self.input["wb_roi_indexes"][1],
        self.input["wb_roi_indexes"][2]] = tmp_output[self.input["wb_roi_indexes"][0],
        self.input["wb_roi_indexes"][1],
        self.input["wb_roi_indexes"][2]]
        self.rtqa_volume[:, :, :, 0] = output_vol.reshape(self.input["dim"], order="F")

        if not self.input["is_auto_rtqa"]:
            tmp_output, self.volume_data["mean_bas_vol"], self.volume_data["m2_bas_vol"], \
                self.volume_data["var_bas_vol"], self.volume_data["mean_cond_vol"], self.volume_data["m2_cond_vol"], \
                self.volume_data["var_cond_vol"] = self.cnr(self.volume_data["mean_bas_vol"],
                                                            self.volume_data["m2_bas_vol"],
                                                            self.volume_data["var_bas_vol"],
                                                            self.volume_data["mean_cond_vol"],
                                                            self.volume_data["m2_cond_vol"],
                                                            self.volume_data["var_cond_vol"],
                                                            volume, self.volume_data["iter_bas"],
                                                            self.volume_data["iter_cond"], index_volume)

            output_vol = np.zeros(tmp_output.shape)
            output_vol[self.input["wb_roi_indexes"][0],
            self.input["wb_roi_indexes"][1],
            self.input["wb_roi_indexes"][2]] = tmp_output[self.input["wb_roi_indexes"][0],
            self.input["wb_roi_indexes"][1],
            self.input["wb_roi_indexes"][2]]
            self.rtqa_volume[:, :, :, 1] = output_vol.reshape(self.input["dim"], order="F")

            if index_volume in self.ind_bas:
                self.volume_data["iter_bas"] += 1
            if index_volume in self.ind_cond:
                self.volume_data["iter_cond"] += 1

        self.input["rtqa_vol_ready"] = True

    # --------------------------------------------------------------------------
    def calculate_rtqa_ts(self, index_volume):

        for roi in range(self.nr_rois):

            data = self.ts_data[0, index_volume, roi]

            # AR(1) was not applied.
            self.r_snr[roi, index_volume], \
                self.r_mean[roi, index_volume], \
                self.m2[roi, index_volume], \
                self.r_var[roi, index_volume] = self.snr(self.r_mean[roi, index_volume - 1],
                                                         self.m2[roi, index_volume - 1],
                                                         data, self.block_iter)

            # GLM regressors were estimated for time-series with AR(1) applied
            if self.ts_data[6, :, roi].any():
                data_noreg = self.ts_data[6, index_volume, roi]
                self.r_no_reg_SNR[roi, index_volume], self.r_no_reg_mean[roi, index_volume], \
                    self.no_reg_m2[roi], \
                    self.r_no_reg_var[roi, index_volume] = self.snr(self.r_no_reg_mean[roi, index_volume - 1],
                                                                    self.no_reg_m2[roi],
                                                                    data_noreg, self.block_iter)

            if not self.input["is_auto_rtqa"]:
                self.r_cnr[roi, index_volume], self.mean_bas[roi, index_volume], \
                    self.m2_bas[roi], self.var_bas[roi, index_volume], \
                    self.mean_cond[roi, index_volume], self.m2_cond[roi], \
                    self.var_cond[roi, index_volume] = self.cnr(self.mean_bas[roi, index_volume - 1], self.m2_bas[roi],
                                                                self.var_bas[roi, index_volume - 1],
                                                                self.mean_cond[roi, index_volume - 1],
                                                                self.m2_cond[roi], self.var_cond[roi, index_volume - 1],
                                                                data, self.iter_bas, self.iter_cond, index_volume)

        self.block_iter += 1
        if not self.input["is_auto_rtqa"]:
            if index_volume in self.ind_bas:
                self.iter_bas += 1
            if index_volume in self.ind_cond:
                self.iter_cond += 1

        data_glm = np.array(self.ts_data[5, index_volume, :], ndmin=2)
        data_proc = np.array(self.ts_data[1, index_volume, :], ndmin=2)
        data_pos_spikes = self.input["pos_spikes"]
        data_neg_spikes = self.input["neg_spikes"]

        self.calculate_spikes(data_glm, index_volume, data_pos_spikes, data_neg_spikes)
        self.calculate_mse(index_volume, data_glm.T, data_proc.T)

    # --------------------------------------------------------------------------
    def snr(self, rMean, m2, data, blockIter):
        """ Recursive SNR calculation

        :param rMean: previous mean value of input data
        :param m2: ptrvious squared mean difference of input data
        :param data: input data
        :param blockIter: iteration number
        :return: calculated rSNR, rMean, rM2 and rVariance
        """

        if blockIter:

            prevMean = rMean
            rMean = prevMean + (data - prevMean) / (blockIter + 1)
            m2 = m2 + (data - prevMean) * (data - rMean)
            rVar = m2 / blockIter
            rSNR = rMean / (rVar ** (.5))
            blockIter += 1

        else:

            rMean = data
            m2 = np.zeros(data.shape, order="F")
            rVar = np.zeros(data.shape, order="F")
            rSNR = np.zeros(data.shape, order="F")
            blockIter = 1

        if not isinstance(data, np.ndarray) and blockIter < 8:

            rSNR = 0

        return rSNR, rMean, m2, rVar,

    # --------------------------------------------------------------------------
    def cnr(self, meanBas, m2Bas, varBas, meanCond, m2Cond, varCond, data, iterBas, iterCond, indexVolume):
        """ Recursive time-series CNR calculation

        :param data: new value of raw time-series
        :param indexVolume: current volume index
        :param isNewDCMBlock: flag of new dcm block
        :return: calculated rCNR, rMeans, rM2s and rVariances
        """

        if indexVolume in self.ind_bas:
            if not iterBas:
                meanBas = data
                m2Bas = np.zeros(data.shape, order="F")
                varBas = np.zeros(data.shape, order="F")
                iterBas += 1

            else:
                prevMeanBas = meanBas
                meanBas = meanBas + (data - meanBas) / (iterBas + 1)
                m2Bas = m2Bas + (data - prevMeanBas) * (data - meanBas)
                varBas = m2Bas / iterBas
                iterBas += 1

        if indexVolume in self.ind_cond:
            if not iterCond:
                meanCond = data
                m2Cond = np.zeros(data.shape, order="F")
                varCond = np.zeros(data.shape, order="F")
                iterCond += 1
            else:
                prevMeanCond = meanCond
                meanCond = meanCond + (data - meanCond) / (iterCond + 1)
                m2Cond = m2Cond + (data - prevMeanCond) * (data - meanCond)
                varCond = m2Cond / iterCond
                iterCond += 1

        if iterCond:
            rCNR = (meanCond - meanBas) / (np.sqrt(varCond + varBas))
        else:
            rCNR = np.zeros(data.shape, order="F")

        return rCNR, meanBas, m2Bas, varBas, meanCond, m2Cond, varCond

    # --------------------------------------------------------------------------
    def _di(self, i):
        return np.array(self.mc_params[i][0:3])

    # --------------------------------------------------------------------------
    def _ri(self, i):
        return np.array(self.mc_params[i][3:6])

    # --------------------------------------------------------------------------
    def _ij_FD(self, i, j):  # displacement from i to j
        return sum(np.absolute(self._di(j) - self._di(i))) + \
            sum(np.absolute(self._ri(j) - self._ri(i))) * self.radius

    # --------------------------------------------------------------------------
    def all_fd(self):
        i = len(self.mc_params) - 1

        if self.input["iteration"] == 0:
            self.fd = np.append(self.fd, 0)
            self.mean_fd = 0

        else:
            self.fd = np.append(self.fd, self._ij_FD(i - 1, i))
            self.mean_fd = self.mean_fd + (self.fd[-1] - self.mean_fd) / (self.block_iter + 1)

        if self.fd[-1] >= self.threshold[1]:
            self.exc_fd[0] += 1

            if self.exc_fd_indexes_1[-1] == -1:
                self.exc_fd_indexes_1 = np.array([i - 1])
            else:
                self.exc_fd_indexes_1 = np.append(self.exc_fd_indexes_1, i - 1)

            if self.fd[-1] >= self.threshold[2]:
                self.exc_fd[1] += 1

                if self.exc_fd_indexes_2[-1] == -1:
                    self.exc_fd_indexes_2 = np.array([i - 1])
                else:
                    self.exc_fd_indexes_2 = np.append(self.exc_fd_indexes_2, i - 1)

    # --------------------------------------------------------------------------
    def micro_displacement(self):

        n = len(self.mc_params) - 1
        sqr_displ = 0

        if self.input["iteration"] == 0:
            self.md = np.append(self.md, 0)
            self.mean_md = 0

        else:
            for i in range(3):
                sqr_displ += self.mc_params[n, i] ** 2

            self.rsq_displ = np.append(self.rsq_displ, np.sqrt(sqr_displ))

            self.md = np.append(self.md, abs(self.rsq_displ[-2] - self.rsq_displ[-1]))
            self.mean_md = self.mean_md + (self.md[-1] - self.mean_md) / (self.block_iter + 1)

        if self.md[-1] >= self.threshold[0]:
            self.exc_md += 1
            if self.exc_md_indexes[-1] == -1:
                self.exc_md_indexes = np.array([n - 1])
            else:
                self.exc_md_indexes = np.append(self.exc_md_indexes, n - 1)

    # --------------------------------------------------------------------------
    def calc_mc(self):

        if self.input["iteration"] == 0:
            self.output["mc_offset"] = self.input["offset_mc"]
            self.mc_params = self.output["mc_offset"]
        else:
            self.mc_params = np.vstack((self.mc_params, self.input["mc_ts"]))
        self.micro_displacement()
        self.all_fd()

    # --------------------------------------------------------------------------
    def calculate_spikes(self, data, indexVolume, posSpikes, negSpikes):
        """ Spikes and GLM signal recording

        :param data: signal values after GLM process
        :param indexVolume: current volume index
        :param posSpikes: flags of positive spikes
        :param negSpikes: flags of negative spikes
        """

        sz, l = data.shape
        self.glm_proc_time_series[:, indexVolume] = data.T.squeeze()

        for i in range(sz):
            if posSpikes[i] == 1:
                if self.pos_spikes[str(i)].any():
                    self.pos_spikes[str(i)] = np.append(self.pos_spikes[str(i)], indexVolume)
                else:
                    self.pos_spikes[str(i)] = np.array([indexVolume])
            if negSpikes[i] == 1 and l > 2:
                if self.neg_spikes[str(i)].any():
                    self.neg_spikes[str(i)] = np.append(self.neg_spikes[str(i)], indexVolume)
                else:
                    self.neg_spikes[str(i)] = np.array([indexVolume])

    # --------------------------------------------------------------------------
    def calculate_mse(self, indexVolume, inputSignal, outputSignal):
        """ Low pass filter performance estimated by recursive mean squared error

        :param indexVolume: current volume index
        :param inputSignal: signal value before filtration
        :param outputSignal: signal value after filtration

        """

        sz = inputSignal.size
        n = self.block_iter

        for i in range(sz):
            self.r_mse[i, indexVolume] = (n / (n + 1)) * self.r_mse[i, indexVolume - 1] + (
                    (inputSignal[i] - outputSignal[i]) ** 2) / (n + 1)

    # --------------------------------------------------------------------------
    def calculate_dvars(self, volume, index_volume):

        if self.prev_vol.size == 0:
            dvars_diff = (volume / self.input["dvars_scale"]) ** 2
        else:
            dvars_diff = ((self.prev_vol - volume) / self.input["dvars_scale"]) ** 2
        dvars_value = 100 * (np.mean(dvars_diff, axis=None)) ** .5

        self.prev_vol = volume

        if index_volume == 0:
            self.dvars = np.append(self.dvars, 0)
        else:
            self.dvars = np.append(self.dvars, dvars_value)

        if self.dvars[-1] > self.default_dvars_threshold:
            self.exc_dvars = self.exc_dvars + 1

    # --------------------------------------------------------------------------
    def rtqa_data_save(self):
        """ Packaging of python RTQA data for following save
        """

        logger.info("Saving rtQA data to rtQA.mat")

        tsRTQA = dict.fromkeys(['rMean', 'rVar', 'rSNR', 'rNoRegSNR',
                                'meanBas', 'varBas', 'meanCond', 'varCond', 'rCNR',
                                'excFDIndexes_1', 'excFDIndexes_2', 'excMDIndexes',
                                'FD', 'MD', 'DVARS', 'rMSE'])

        tsRTQA['rMean'] = self.output["rMean"]
        tsRTQA['rVar'] = self.output["rVar"]
        tsRTQA['rSNR'] = self.output["rSNR"]
        tsRTQA['rNoRegSNR'] = self.output["rNoRegSNR"]
        tsRTQA['meanBas'] = self.output["meanBas"]
        tsRTQA['varBas'] = self.output["varBas"]
        tsRTQA['meanCond'] = self.output["meanCond"]
        tsRTQA['varCond'] = self.output["varCond"]
        tsRTQA['rCNR'] = self.output["rCNR"]
        tsRTQA['excFDIndexes_1'] = self.exc_fd_indexes_1
        tsRTQA['excFDIndexes_2'] = self.exc_fd_indexes_2
        tsRTQA['excMDIndexes'] = self.exc_md_indexes
        tsRTQA['FD'] = self.output["FD"]
        tsRTQA['MD'] = self.output["MD"]
        tsRTQA['DVARS'] = self.output["DVARS"]
        tsRTQA['rMSE'] = self.output["rMSE"]

        savemat("rtQA.mat", tsRTQA)
