# -*- coding: utf-8 -*-
import shutil
import time
from pathlib import Path
from opennft.filewatcher import FileWatcher


# --------------------------------------------------------------------------
def main():
    # config: https://github.com/OpenNFT/pyOpenNFT/pull/9

    # hardcode config for test
    path = Path('C:/work/pyOpenNFT/tests/data')

    path = path / 'fw_test' / 'dcm'
    fw = FileWatcher()
    fw.start_watching(False, path, "001_000007_000001.dcm", "001_000007_000001.dcm", file_ext="dcm")
    fn = next(fw)
    print(fn)
    print(fn.parts[-1])
    assert fn.parts[-1] == '001_000007_000001.dcm', "Error obtaining the next file"


# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()
