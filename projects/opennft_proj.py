# -*- coding: utf-8 -*-
import shutil
import time
from pathlib import Path
from loguru import logger

from opennft.filewatcher import FileWatcher
from opennft.nfbcalc import Nfb
from opennft import LegacyNftConfigLoader

import opennft.nftsession as nftsession


# --------------------------------------------------------------------------
def main():
    # config: https://github.com/OpenNFT/pyOpenNFT/pull/9

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

    iteration = nftsession.NftIteration(session)
    nfb_calc = Nfb(session, iteration)

    fw = FileWatcher()
    fw_path = Path(config.watch_dir)
    fw.start_watching(False, fw_path, config.first_file_name, config.first_file_name, file_ext="dcm")

    for vol_filename in fw:
        # main loop iteration

        logger.info(f"Got scan file: {vol_filename}")

        if iteration.iter_number == 0:
            logger.info(f"First volume initialization")
            # do some first volume setup

        if iteration.pre_iter < iteration.iter_number:
            # pre-acquisition routine
            nfb_calc.main_loop_entry()

        iteration.load_vol(vol_filename, "dcm")

        iteration.pre_iter = iteration.iter_number

        if iteration.iter_number < session.config.skip_vol_nr:
            logger.info(f"Scan file skipped")
            iteration.iter_number += 1
            continue

        time_start = time.time()
        iteration.process_vol()
        iteration.process_time_series()

        iteration.iter_number += 1
        elapsed_time = time.time() - time_start

        logger.info('{} {:.4f} {}', "Elapsed time: ", elapsed_time, 's')

    iteration.save_time_series()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
