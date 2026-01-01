import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer

from login_page import LoginPage
from selection_page import SelectionPage
from gui import OCRGui
from gui_live import OCRLiveGui


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Application Suite")
        self.showMaximized()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_page = LoginPage()
        self.selection_page = SelectionPage()

        self.ocr_config_page = None
        self.ocr_live_page = None

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.selection_page)

        self.login_page.login_success.connect(self.show_selection)
        self.selection_page.config_selected.connect(self.open_ocr_config)
        self.selection_page.live_selected.connect(self.open_ocr_live)
        self.selection_page.logout_clicked.connect(self.logout)

        self.stack.setCurrentWidget(self.login_page)

    # ---------------- NAVIGATION ----------------
    def show_selection(self):
        self.stack.setCurrentWidget(self.selection_page)

    def open_ocr_config(self):
        # Step 1: let UI breathe
        self.stack.setCurrentWidget(self.selection_page)

        # Step 2: defer heavy creation
        QTimer.singleShot(0, self._create_ocr_config)

    def open_ocr_live(self):
        self.stack.setCurrentWidget(self.selection_page)
        QTimer.singleShot(0, self._create_ocr_live)

    # ---------------- DEFERRED CREATORS ----------------
    def _create_ocr_config(self):
        if self.ocr_config_page is None:
            self.ocr_config_page = OCRGui()
            self.ocr_config_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.ocr_config_page)

        self.stack.setCurrentWidget(self.ocr_config_page)

    def _create_ocr_live(self):
        if self.ocr_live_page is None:
            self.ocr_live_page = OCRLiveGui()
            self.ocr_live_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.ocr_live_page)

        self.stack.setCurrentWidget(self.ocr_live_page)

    def logout(self):
        self.stack.setCurrentWidget(self.login_page)
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()
        self.login_page.error_label.clear()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    app.setFont(QFont("Segoe UI", 10))
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())