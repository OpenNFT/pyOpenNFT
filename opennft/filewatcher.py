import os
import fnmatch
import glob
import queue
import re
from pathlib import Path
from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class NewFileEventHandler(FileSystemEventHandler):
    def __init__(self, filepat, fq: queue.Queue, recorder=None):
    # TODO Fix erd
    # def __init__(self, filepat, fq: queue.Queue, recorder: erd.EventRecorder=None):
        self.filepat = filepat
        self.fq = fq
        self.recorder = recorder

    def on_created(self, event):
        # if not event.is_directory and event.src_path.endswith(self.filepat):
        if not event.is_directory and fnmatch.fnmatch(os.path.basename(event.src_path), self.filepat):
            if not self.recorder is None:
                pass
                # t1
                # FIX self.recorder.recordEvent(erd.Times.t1, 0, time.time())
            self.fq.put(event.src_path)


class FileWatcher():
    def __init__(self):
        self.watch_folder = None
        self.first_filename = None
        self.files_queue = None
        self.file_ext = None
        self.first_filename_txt = None
        self.fs_observer = None
        self.search_string = None
        self.online = None
    # --------------------------------------------------------------------------

    def __iter__(self):
        return self
    # --------------------------------------------------------------------------

    def __next__(self) -> Path:
        # TODO find all files_queue in old version!!!
        fname = None
        try:
            fname = self.files_queue.get_nowait()
        except queue.Empty:
            fname = None

        return Path(fname)
    # --------------------------------------------------------------------------

    def get_search_string(self, file_name_template, path, ext) -> str:
        file_series_part = re.findall(r"\{#:(\d+)\}", file_name_template)
        file_num_part = re.findall(r"_\d+_(\d+.\w+)", file_name_template)
        if len(file_series_part) > 0:
            file_series_len = int(file_series_part[0])
            fname = os.path.splitext(os.path.basename(path))[0][:-file_series_len]
            search_string = '%s*%s' % (fname, ext)
        elif len(file_num_part) > 0:
            fname = file_name_template.replace(file_num_part[0], "*")
            search_string = '%s%s' % (fname, ext)
        else:
            search_string = '*%s' % ext

        return search_string
    # --------------------------------------------------------------------------

    def start_watching(self, online, watch_folder, first_filename, first_filename_txt, file_ext=None,
                       event_recorder=None):
        self.watch_folder = watch_folder
        self.first_filename = first_filename
        self.files_queue = queue.Queue()
        self.recorder = event_recorder
        self.online = online

        # path = os.path.join(self.P['WatchFolder'], self.P['FirstFileName'])
        path = os.path.join(watch_folder, first_filename)

        ext = re.findall(r"\.\w*$", str(path))
        if not ext:
            ext = file_ext
            # TODO move it to the call
            # if self.P['DataType'] == 'IMAPH':
            #     ext = config.IMAPH_FILES_EXTENSION
            # else:  # dicom as default
            #    ext = config.DICOM_FILES_EXTENSION
        else:
            ext = ext[-1]

        self.file_ext = ext
        self.first_filename_txt = first_filename_txt
        # self.searchString = self.getFileSearchString(self.P['FirstFileNameTxt'], path, ext)
        self.search_string = self.get_search_string(first_filename_txt, path, ext)
        
        if online:
            path = os.path.dirname(path)
            logger.info('Online watching for {} in {}', self.search_string, path)
            event_handler = NewFileEventHandler(self.search_string, self.files_queue, self.recorder)

            self.fs_observer = Observer()
            self.fs_observer.schedule(event_handler, path, recursive=True)
            self.fs_observer.start()
        else:       
            path = os.path.join(os.path.dirname(path), self.search_string)
            logger.info('Offline searching for {}', path)
            files = sorted(glob.glob(path))
        
            if not files:
                logger.info("No files found in offline mode. Check WatchFolder settings!")
                # self.stop()
                return

            self.files_queue = queue.Queue()

            for f in files:
                self.files_queue.put(f)

        # TODO move to the call
        # self.call_timer.start()
    # --------------------------------------------------------------------------


if __name__ == '__main__':
    pass


