# -*- coding: utf-8 -*-
import shutil
import time
from pathlib import Path
from loguru import logger
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

from opennft.filewatcher import FileWatcher
from opennft.nfbcalc import Nfb
from opennft import LegacyNftConfigLoader

import opennft.nftsession as nftsession
from opennft.config import config as con


class OpenNFTCalc(mp.Process):

    def __init__(self, service_dict):

        super().__init__()
        self._service_data = service_dict
        self.init_data()

    def init_data(self):

        config_path = Path().resolve()
        config_path /= 'config.ini'

        loader = LegacyNftConfigLoader()
        loader.load(config_path)

        config = loader.config  # LegacyNftConfig instance
        simulation_protocol = loader.simulation_protocol  # simulation protocol dictionary from JSON

        session = nftsession.NftSession(config)
        session.setup()
        session.set_protocol(simulation_protocol)
        # setup ROIs for session
        # setup mr_reference for session

        self.config = config
        self.simulation_protocol = simulation_protocol
        self.session = session
        self.iteration = nftsession.NftIteration(session)
        self.nfb_calc = Nfb(session, self.iteration)

        self._service_data["nr_vol"] = self.session.config.volumes_nr-self.session.config.skip_vol_nr
        self._service_data["nr_rois"] = self.session.nr_rois

    def init_shm(self):

        nr_vol = self.session.config.volumes_nr-self.session.config.skip_vol_nr
        nr_rois = self.session.nr_rois

        self.mc_shm = shared_memory.SharedMemory(name=con.shm_file_names[0])
        self.mc_data = np.ndarray(shape=(nr_vol, 6), dtype=np.float32, buffer=self.mc_shm.buf)

    # --------------------------------------------------------------------------
    def run(self):
        # config: https://github.com/OpenNFT/pyOpenNFT/pull/9

        self.init_shm()
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
                self.nfb_calc.main_loop_entry()

            self.iteration.load_vol(vol_filename, "dcm")

            self.iteration.pre_iter = self.iteration.iter_number

            if self.iteration.iter_number < self.session.config.skip_vol_nr:
                logger.info(f"Scan file skipped")
                self.iteration.iter_number += 1
                continue

            self._service_data["init"] = (self.iteration.iter_number == self.session.config.skip_vol_nr)

            time_start = time.time()
            self.iteration.process_vol()

            self.mc_data[self.iteration.iter_norm_number, :] = self.iteration.mr_time_series.mc_params[:,-1].T
            self._service_data["data_ready_flags"] = True

            self.iteration.process_time_series()

            self.iteration.iter_number += 1
            elapsed_time = time.time() - time_start

            logger.info('{} {:.4f} {}', "Elapsed time: ", elapsed_time, 's')

        # self.iteration.save_time_series()

        self.mc_shm.close()
        print("calc process finished")

#
# # --------------------------------------------------------------------------
# if __name__ == '__main__':
#     main()
