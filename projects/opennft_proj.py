# -*- coding: utf-8 -*-
import shutil
import time
from pathlib import Path
from loguru import logger

from opennft.filewatcher import FileWatcher
import opennft.nft_classes_stub as nft

import opennft.nftsession as nftsession
import opennft.mrvol as mrvol
from opennft import LegacyNftConfigLoader


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
    # setup ROIs for session
    # setup mr_reference for session

    iteration = nftsession.NftIteration(session)

    fw = FileWatcher()
    fw_path = Path(config.watch_dir)
    fw.start_watching(False, fw_path, config.first_file_name, config.first_file_name, file_ext="dcm")

    for vol_filename in fw:
        # main loop iteration

        logger.info(f"Got scan file: {vol_filename}")

        if iteration.iter_number == 0:
            logger.info(f"First volume initialization")
            # do some first volume setup

        iteration.load_vol(vol_filename, "dcm")
        if iteration.iter_number < session.config.skip_vol_nr:
            break
        
        iteration.process_vol()
        iteration.process_time_series()

        iteration.iter_number += 1

    iteration.save_time_series()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
