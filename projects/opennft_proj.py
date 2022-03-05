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

    #path = path / 'fw_test' / 'dcm'

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

        # iteration.handlers_data['load_scan']['load_mr_vol'] = [iteration.mr_vol, vol_filename]
        # iteration.dispatch_handlers()

        # or without handlers:
        iteration.load_vol(vol_filename, "dcm")
        iteration.process_vol()

        iteration.iter_number += 1


# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
