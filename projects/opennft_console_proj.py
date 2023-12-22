# -*- coding: utf-8 -*-

import time
from pathlib import Path
from loguru import logger
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
# from pyniexp.connection import Udp
from rtspm import spm_setup

from opennft.filewatcher import FileWatcher
from opennft.nfbcalc import Nfb
from opennft import LegacyNftConfigLoader
from opennft.utils import zscore
from opennft import eventrecorder as erd

import opennft.nftsession as nftsession
from opennft.config import config as con


class OpenNFTCoreProj(mp.Process):

    # --------------------------------------------------------------------------
    def __init__(self, service_dict):

        super().__init__()

        self.session = None
        self.iteration = None
        self.config = None
        self.simulation_protocol = None
        self.nfb_calc = None
        # self.udp_sender = None
        # self.udp_send_condition = False

        self.exchange_data = service_dict
        self.init_data()
        self.init_exchange_data()
        # if self.exchange_data["use_udp_feedback"] is None:
        #     self.use_udp_feedback = self.config.use_udp_feedback
        #     self.exchange_data['offline'] = self.config.offline_mode
        #     self.exchange_data["use_udp_feedback"] = self.config.use_udp_feedback
        # else:
        #     self.use_udp_feedback = self.exchange_data["use_udp_feedback"]
        self.recorder = erd.EventRecorder()
        self.recorder.initialize(self.config.volumes_nr)

    # --------------------------------------------------------------------------
    def init_data(self):

        config_path = Path().resolve()
        config_path /= self.exchange_data["set_file"]

        loader = LegacyNftConfigLoader()
        loader.load(config_path)

        config = loader.config  # LegacyNftConfig instance
        simulation_protocol = loader.simulation_protocol  # simulation protocol dictionary from JSON

        session = nftsession.NftSession(config)
        # setup ROIs for session
        # setup mr_reference for session
        session.setup()
        session.set_protocol(simulation_protocol)
        session.spm = spm_setup(config.tr,
                                config.volumes_nr - config.skip_vol_nr,
                                np.mean(session.reference_vol.volume, axis=None),
                                session.offsets,
                                session.first_nf_inds,
                                session.prot_names
                                )

        self.session = session
        self.iteration = nftsession.NftIteration(session)

        self.iteration.lin_regr = zscore(
            np.array(range(0, config.volumes_nr - config.skip_vol_nr), ndmin=2).transpose())

        self.config = config
        self.simulation_protocol = simulation_protocol
        self.nfb_calc = Nfb(session, self.iteration)

    # --------------------------------------------------------------------------
    def init_exchange_data(self):

        self.exchange_data["nr_vol"] = self.session.config.volumes_nr - self.session.config.skip_vol_nr
        self.exchange_data["nr_rois"] = self.session.nr_rois
        self.exchange_data["vol_dim"] = self.session.reference_vol.dim
        self.exchange_data["mosaic_dim"] = self.session.img2d_dimy, self.session.img2d_dimx
        self.exchange_data["vol_mat"] = self.session.reference_vol.mat
        self.exchange_data["roi_names"] = self.session.roi_names
        self.exchange_data["iter_norm_number"] = self.iteration.iter_norm_number

    # --------------------------------------------------------------------------
    def init_shmem(self):

        nr_vol = self.session.config.volumes_nr - self.session.config.skip_vol_nr
        nr_rois = self.session.nr_rois

        self.mc_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[0])
        self.mc_data = np.ndarray(shape=(nr_vol, 6), dtype=np.float32, buffer=self.mc_shmem.buf)

        self.ts_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[8])
        self.ts_data = np.ndarray(shape=(5, nr_vol, nr_rois), dtype=np.float32, buffer=self.ts_shmem.buf)

        self.nfb_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[9])
        self.nfb_data = np.ndarray(shape=(1, nr_vol), dtype=np.float32, buffer=self.nfb_shmem.buf)

        self.epi_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[2])
        self.epi_volume = np.ndarray(shape=self.session.reference_vol.dim, dtype=np.float32, buffer=self.epi_shmem.buf,
                                     order="F")

        stat_dim = tuple(self.session.reference_vol.dim) + (2,)
        self.stat_shmem = shared_memory.SharedMemory(name=con.shmem_file_names[3])
        self.stat_volume = np.ndarray(shape=stat_dim, dtype=np.float32, buffer=self.stat_shmem.buf, order="F")

    # --------------------------------------------------------------------------
    # def init_udp_sender(self, udp_feedback_ip, udp_feedback_port, udp_feedback_controlchar, udp_send_condition):
    #
    #     self.udp_send_condition = udp_send_condition
    #
    #     if not self.use_udp_feedback:
    #         return
    #
    #     self.udp_sender = Udp(
    #         IP=udp_feedback_ip,
    #         port=udp_feedback_port,
    #         control_signal=udp_feedback_controlchar,
    #         encoding='UTF-8'
    #     )
    #     self.udp_sender.connect_for_sending()
    #     self.udp_sender.sending_time_stamp = True
    #
    #     self.udp_cond_for_contrast = self.session.prot_names
    #     if self.udp_cond_for_contrast[0] != 'BAS':
    #         self.udp_cond_for_contrast.insert(0, 'BAS')

    # --------------------------------------------------------------------------
    # def finalize_udp_sender(self):
    #     if not self.use_udp_feedback:
    #         return
    #     if self.udp_sender is not None:
    #         self.udp_sender.close()
    #     self.use_udp_feedback = False

    # --------------------------------------------------------------------------
    def run(self):
        # config: https://github.com/OpenNFT/pyOpenNFT/pull/9

        # if self.use_udp_feedback:
        #     self.init_udp_sender(self.exchange_data['udp_feedback_ip'],
        #                          self.exchange_data['udp_feedback_port'],
        #                          self.exchange_data['udp_feedback_controlchar'],
        #                          self.exchange_data['udp_send_condition'])

        if con.use_gui:
            self.init_shmem()
        print("calc process started")

        fw = FileWatcher()
        fw_path = Path(self.config.watch_dir)
        fw.start_watching(not self.exchange_data['offline'], fw_path, self.config.first_file_name,
                          self.config.first_file_name,
                          file_ext="dcm", event_recorder=self.recorder)

        for vol_filename in fw:
            # main loop iteration

            # if self.exchange_data['close_udp']:
            #     self.finalize_udp_sender()

            if not self.exchange_data['offline']:
                while vol_filename is None:
                    time.sleep(1)
                    break
                if vol_filename is None:
                    logger.info('Waiting for a file...')
                    continue
            elif vol_filename is None:
                break

            logger.info(f"Got scan file: {vol_filename}")

            if self.iteration.iter_number == 0:
                logger.info(f"First volume initialization")
                # do some first volume setup

            if self.iteration.pre_iter < self.iteration.iter_number:
                # pre-acquisition routine
                self.iteration.iter_norm_number = self.iteration.iter_number - self.session.config.skip_vol_nr
                self.nfb_calc.nfb_init()

                # if self.config.type in ['PSC', 'Corr']:
                #     if self.iteration.iter_number > self.config.skip_vol_nr and self.udp_send_condition:
                #         self.udp_sender.send_data(
                #             self.udp_cond_for_contrast[int(self.nfb_calc.condition - 1)])

                # elif self.config.type == 'SVM':
                #     if self.nfb_calc.display_data and self.use_udp_feedback:
                #         logger.info('Sending by UDP - instrValue = ')  # + str(self.displayData['instrValue'])
                        # self.udp_sender.send_data(self.displayData['instrValue'])
            # t1
            self.recorder.record_event(erd.Times.t1, self.iteration.iter_number + 1, time.time())
            self.iteration.load_vol(vol_filename, "dcm")

            self.iteration.pre_iter = self.iteration.iter_number

            if self.iteration.iter_number < self.session.config.skip_vol_nr:
                logger.info(f"Scan file skipped")
                self.iteration.iter_number += 1
                continue

            self.exchange_data["init"] = (self.iteration.iter_number == self.session.config.skip_vol_nr)
            self.exchange_data["iter_norm_number"] = self.iteration.iter_norm_number

            time_start = time.time()

            # t2
            self.recorder.record_event(erd.Times.t2, self.iteration.iter_number+1, time.time())
            self.iteration.process_vol()
            stat_ready = self.iteration.iglm()

            # t3
            self.recorder.record_event(erd.Times.t3, self.iteration.iter_number+1, time.time())

            if con.use_gui:

                self.epi_volume[:, :, :] = self.iteration.mr_vol.volume
                if stat_ready:
                    self.stat_volume[:, :, :, 0] = self.iteration.iglm_params["stat_map_3d_pos"]
                    self.stat_volume[:, :, :, 1] = self.iteration.iglm_params["stat_map_3d_neg"]
                    self.exchange_data["overlay_ready"] = True

                self.exchange_data["ready_to_form"] = True

            self.iteration.process_time_series()

            # t4
            self.recorder.record_event(erd.Times.t4, self.iteration.iter_number+1, time.time())

            self.nfb_calc.nfb_calc()

            # t5
            self.recorder.record_event(erd.Times.t5, self.iteration.iter_number+1, time.time())

            # if self.nfb_calc.display_data:
            #     if self.use_udp_feedback:
            #         # logger.info('Sending by UDP - dispValue = {}', self.nfb_calc.display_data['disp_value'])
            #         # self.udp_sender.send_data(float(self.nfb_calc.display_data['disp_value']))
            #
            #         cond = self.nfb_calc.condition
            #         if cond == 2:
            #             val = 'N ' + str(self.nfb_calc.display_data["disp_value"])
            #         else:
            #             val = 'B ' + str(self.nfb_calc.display_data["disp_value"])
            #
            #         logger.info('Sending by UDP - dispValue = {}', val)
            #         self.udp_sender.send_data(val)

            iter_number = self.iteration.iter_norm_number

            if con.use_gui:

                self.mc_data[iter_number, :] = self.iteration.mr_time_series.mc_params[:, -1].T
                for i in range(self.session.nr_rois):
                    if self.config.prot != 'InterBlock':
                        self.ts_data[0, iter_number, i] = self.iteration.mr_time_series.disp_raw_time_series[i][
                            iter_number].T
                    else:
                        self.ts_data[0, iter_number, i] = self.iteration.mr_time_series.raw_time_series[i][iter_number]
                    self.ts_data[1, iter_number, i] = self.iteration.mr_time_series.kalman_proc_time_series[i][iter_number]
                    self.ts_data[2, iter_number, i] = self.iteration.mr_time_series.scale_time_series[i][iter_number]
                    self.ts_data[3, iter_number, i] = self.iteration.mr_time_series.output_pos_min[i][iter_number]
                    self.ts_data[4, iter_number, i] = self.iteration.mr_time_series.output_pos_max[i][iter_number]
                self.nfb_data[0, iter_number] = self.nfb_calc.disp_values[iter_number] / self.config.max_feedback_val
                self.exchange_data["data_ready_flag"] = True

            self.exchange_data["elapsed_time"] = time.time() - time_start

            logger.info('{} {:.4f} {}', "Elapsed time: ", self.exchange_data["elapsed_time"], 's')

            # d0
            self.recorder.record_event(erd.Times.d0, self.iteration.iter_number+1, self.exchange_data["elapsed_time"])
            self.iteration.iter_number += 1

            if not self.exchange_data['offline'] and self.iteration.iter_number == self.config.volumes_nr:
                logger.info('Last iteration {} reached...', self.iteration.iter_number)
                fw.stop()
                break

        self.iteration.save_time_series(self.config.work_dir / ('NF_Data_' + str(self.config.nf_run_nr)))
        self.iteration.save_stat_vols(self.config.work_dir / ('NF_Data_' + str(self.config.nf_run_nr)))
        self.nfb_calc.nfb_save(self.config.work_dir / ('NF_Data_' + str(self.config.nf_run_nr)))

        if self.iteration.iter_norm_number > 1:
            path = self.config.work_dir / ('NF_Data_' + str(self.config.nf_run_nr))
            fname = path / ('pyTimeVectors_' + str(self.config.nf_run_nr).zfill(2) + '.txt')
            self.recorder.save_txt(str(fname))

        if con.use_gui:

            self.mc_shmem.close()
            self.epi_shmem.close()
            self.ts_shmem.close()
            self.nfb_shmem.close()
            self.stat_shmem.close()

        self.exchange_data['is_stopped'] = True

        print("calc process finished")
