from PyQt6.QtCore import QThread, pyqtSignal
import requests
import time

#

class RestWorker(QThread):
    finished = pyqtSignal(object)  # Signal to emit the result

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            start = time.time()
            response = requests.get(self.url)
            end = time.time()
            if response.status_code == 200:
                self.finished.emit((response.json(), start, end))
            else:
                self.finished.emit(None)
        except Exception as e:
            self.finished.emit(None)