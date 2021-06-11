from opennft.filewatcher import FileWatcher
from pathlib import Path

def test_fw_1(data_path):
    print(data_path)
    path = data_path / 'fw_test' / 'dcm'
    fw = FileWatcher()
    fw.start_watching(False, path, "001_000007_000001.dcm", "001_000007_000001.dcm", file_ext="dcm")
    fn = Path(next(fw))
    print(fn)
    print(fn.parts[-1])
    if fn.parts[-1] == '001_000007_000001.dcm':
        assert True, "Done"
    else:
        assert False, "Error obtaining the next file"
