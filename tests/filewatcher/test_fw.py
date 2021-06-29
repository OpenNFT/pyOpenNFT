import time
import shutil
from pathlib import Path
from opennft.filewatcher import FileWatcher


def test_online(data_path):
    path = data_path / 'fw_test'
    path_online = path / 'online'

    for f in path_online.glob('*.dcm'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    fw = FileWatcher()
    fw.start_watching(True, path / 'online', "001_000007_000001.dcm", "001_000007_000001.dcm", file_ext="dcm")
    src = path / 'dcm' / "001_000007_000001.dcm"
    dst = path / 'online' / "001_000007_000001.dcm"
    shutil.copy(src, dst)
    fn = next(fw)
    i = 0
    while fn is None:
        time.sleep(1)
        fn = next(fw)
        i += 1
        if i > 100:
            break
    if fn is not None:
        print(fn)
        assert fn.parts[-1] == '001_000007_000001.dcm', "Error obtaining the next file"
    else:
        assert False, "Error obtaining the next file"


def test_offline(data_path):
    print(data_path)
    path = data_path / 'fw_test' / 'dcm'
    fw = FileWatcher()
    fw.start_watching(False, path, "001_000007_000001.dcm", "001_000007_000001.dcm", file_ext="dcm")
    fn = next(fw)
    print(fn)
    print(fn.parts[-1])
    assert fn.parts[-1] == '001_000007_000001.dcm', "Error obtaining the next file"
