import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from gui import OCRGui          # existing config UI
from gui_live import OCRLiveGui        # new live UI


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Application")
        self.showMaximized()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: OCR Configuration
        self.ocr_config_tab = OCRGui()
        self.tabs.addTab(self.ocr_config_tab, "OCR Configuration")

        # Tab 2: OCR Live
        self.ocr_live_tab = OCRLiveGui()
        self.tabs.addTab(self.ocr_live_tab, "OCR Live")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
