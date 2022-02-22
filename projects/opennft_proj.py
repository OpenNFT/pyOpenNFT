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

    # volume clas creation
    # volume = Volume()
    # ... and initialization
    # volume.set_mc_template(path/to/template)

    iter_number = 0
    for volume_filename in fw:
        # main loop iteration

        logger.info(f"Got scan file: {volume_filename}")

        if iter_number == 0:
            logger.info(f"First volume initialization")
            # do some first volume setup with volume object

        # possible procedure way
        # volume = load_volume(volume_filename)
        # volume = realign_volume(volume)
        # volume = reslice_volume(volume)
        # volume = smooth_volume(volume)

        # better class-based way
        # volume.load_volume(volume_filename)
        # volume.realign_volume()
        # volume.reslice_volume()
        # volume.smooth_volume()
        iter_number += 1

# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
