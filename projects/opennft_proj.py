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

    # hardcode config for test
    # filename = 'config.ini'
    filename = 'C:/work/pyOpenNFT/projects/config.ini'

    loader = LegacyNftConfigLoader()
    loader.load(filename)

    config = loader.config  # LegacyNftConfig instance
    simulation_protocol = loader.simulation_protocol  # simulation protocol dictionary from JSON

    path = Path(config.watch_dir)
    path = path / 'fw_test' / 'dcm'

    config = nft.Config()
    session = nftsession.NftSession(config)
    # setup ROIs for session
    # setup mr_reference for session

    iteration = nftsession.NftIteration(session)

    fw = FileWatcher()
    fw.start_watching(False, path, "001_000007_000001.dcm", "001_000007_000001.dcm", file_ext="dcm")

    for vol_filename in fw:
        # main loop iteration

        logger.info(f"Got scan file: {vol_filename}")

        if iteration.iter_number == 0:
            logger.info(f"First volume initialization")
            # do some first volume setup

        # iteration.handlers_data['load_scan']['load_mr_vol'] = [iteration.mr_vol, vol_filename]
        # iteration.dispatch_handlers()

        # or without handlers:
        iteration.mr_vol.load_vol(vol_filename)

        iteration.process_vol()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
