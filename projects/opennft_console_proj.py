# -*- coding: utf-8 -*-

import time
from pathlib import Path
from loguru import logger
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from rtspm import spm_setup

from opennft.filewatcher import FileWatcher
from opennft.nfbcalc import Nfb
from opennft import LegacyNftConfigLoader
from opennft.utils import ar_regr, zscore

import opennft.nftsession as nftsession
from opennft.config import config as con


class OpenNFTCoreProj(mp.Process):

    def __init__(self, service_dict):

        super().__init__()

        self.session = None
        self.iteration = None
        self.config = None
        self.simulation_protocol = None
        self.nfb_calc = None

        self.exchange_data = service_dict
        self.init_data()
        self.init_exchange_data()

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

    def init_exchange_data(self):

        self.exchange_data["nr_vol"] = self.session.config.volumes_nr - self.session.config.skip_vol_nr
        self.exchange_data["nr_rois"] = self.session.nr_rois
        self.exchange_data["vol_dim"] = self.session.reference_vol.dim
        self.exchange_data["mosaic_dim"] = self.session.img2d_dimy, self.session.img2d_dimx
        self.exchange_data["vol_mat"] = self.session.reference_vol.mat
        self.exchange_data["roi_names"] = self.session.roi_names
        self.exchange_data["iter_norm_number"] = self.iteration.iter_norm_number

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
    def run(self):
        # config: https://github.com/OpenNFT/pyOpenNFT/pull/9

        self.init_shmem()
        print("calc process started")

        fw = FileWatcher()
        fw_path = Path(self.config.watch_dir)
        fw.start_watching(False, fw_path, self.config.first_file_name, self.config.first_file_name, file_ext="dcm")

        for vol_filename in fw:
            # main loop iteration

            logger.info(f"Got scan file: {vol_filename}")

            if self.iteration.iter_number == 0:
                logger.info(f"First volume initialization")
                # do some first volume setup

            if self.iteration.pre_iter < self.iteration.iter_number:
                # pre-acquisition routine
                self.iteration.iter_norm_number = self.iteration.iter_number - self.session.config.skip_vol_nr
                self.nfb_calc.nfb_init()

            self.iteration.load_vol(vol_filename, "dcm")

            self.iteration.pre_iter = self.iteration.iter_number

            if self.iteration.iter_number < self.session.config.skip_vol_nr:
                logger.info(f"Scan file skipped")
                self.iteration.iter_number += 1
                continue

            self.exchange_data["init"] = (self.iteration.iter_number == self.session.config.skip_vol_nr)
            self.exchange_data["iter_norm_number"] = self.iteration.iter_norm_number

            time_start = time.time()
            self.iteration.process_vol()
            stat_ready = self.iteration.iglm()

            self.epi_volume[:, :, :] = self.iteration.mr_vol.volume
            self.exchange_data["ready_to_form"] = True

            if stat_ready:
                self.stat_volume[:,:,:,0] = self.iteration.iglm_params["stat_map_3d_pos"]
                if self.exchange_data["is_neg"]:
                    self.stat_volume[:,:,:,1] = self.iteration.iglm_params["stat_map_3d_neg"]
                self.exchange_data["overlay_ready"] = True

            self.iteration.process_time_series()
            self.nfb_calc.nfb_calc()

            iter_number = self.iteration.iter_norm_number
            self.mc_data[iter_number, :] = self.iteration.mr_time_series.mc_params[:, -1].T
            for i in range(self.session.nr_rois):
                if self.config.prot != 'InterBlock':
                    self.ts_data[0, iter_number, i] = self.iteration.mr_time_series.disp_raw_time_series[i][iter_number].T
                else:
                    self.ts_data[0, iter_number, i] = self.iteration.mr_time_series.raw_time_series[i][iter_number]
                self.ts_data[1, iter_number, i] = self.iteration.mr_time_series.kalman_proc_time_series[i][iter_number]
                self.ts_data[2, iter_number, i] = self.iteration.mr_time_series.scale_time_series[i][iter_number]
                self.ts_data[3, iter_number, i] = self.iteration.mr_time_series.output_pos_min[i][iter_number]
                self.ts_data[4, iter_number, i] = self.iteration.mr_time_series.output_pos_max[i][iter_number]
            self.nfb_data[0,iter_number] = self.nfb_calc.disp_values[iter_number] / self.config.max_feedback_val
            self.exchange_data["data_ready_flag"] = True

            self.iteration.iter_number += 1
            self.exchange_data["elapsed_time"] = time.time() - time_start

            logger.info('{} {:.4f} {}', "Elapsed time: ", self.exchange_data["elapsed_time"], 's')

        self.iteration.save_time_series()
        self.nfb_calc.nfb_save()

        self.mc_shmem.close()
        self.epi_shmem.close()
        self.ts_shmem.close()
        self.nfb_shmem.close()
        self.stat_shmem.close()

        print("calc process finished")
