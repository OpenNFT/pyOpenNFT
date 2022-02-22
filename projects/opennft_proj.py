# -*- coding: utf-8 -*-
import shutil
import time
from pathlib import Path
from loguru import logger
from opennft.filewatcher import FileWatcher


# --------------------------------------------------------------------------
def main():
    # config: https://github.com/OpenNFT/pyOpenNFT/pull/9

    # hardcode config for test
    path = Path('C:/work/pyOpenNFT/tests/data')
    path = path / 'fw_test' / 'dcm'

    fw = FileWatcher()
    fw.start_watching(False, path, "001_000007_000001.dcm", "001_000007_000001.dcm", file_ext="dcm")

    for volume_filename in fw:
        # main loop iteration

        logger.info(f"Got scan file: {volume_filename}")
        # volume = load_volume()
        # volume = realign_volume(volume)
        # volume = reslice_volume(volume)
        # volume = smooth_volume(volume)



# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
