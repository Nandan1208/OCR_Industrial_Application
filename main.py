import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer

from login_page import LoginPage
from selection_page import SelectionPage
from gui import OCRGui
from gui_live import OCRLiveGui
from gui_barcode import BarcodeGui
from barcode_live_gui import BarcodeLiveGui

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vision System")
        self.showMaximized()

        # ---------------- STACK ----------------
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # ---------------- LIGHT PAGES ----------------
        self.login_page = LoginPage()
        self.selection_page = SelectionPage()

        # ---------------- HEAVY PAGES (LAZY) ----------------
        self.ocr_page = None
        self.live_page = None
        self.barcode_page = None
        self.barcode_live_page = None

        # ---------------- ADD BASE PAGES ----------------
        self.stack.addWidget(self.login_page)      # index 0
        self.stack.addWidget(self.selection_page)  # index 1

        # ---------------- SIGNALS ----------------
        self.login_page.login_success.connect(self.show_selection)

        self.selection_page.config_selected.connect(self.open_ocr)
        self.selection_page.live_selected.connect(self.open_live)
        self.selection_page.barcode_selected.connect(self.open_barcode)
        self.selection_page.barcode_live_selected.connect(self.open_barcode_live)
        self.selection_page.logout_clicked.connect(self.logout)

        # ---------------- START ----------------
        self.stack.setCurrentWidget(self.login_page)

    # =====================================================
    # NAVIGATION
    # =====================================================
    def show_selection(self):
        self.selection_page.set_cards_enabled(True)
        self.stack.setCurrentWidget(self.selection_page)

    def logout(self):
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()
        self.login_page.error_label.clear()
        self.show_selection()
        self.stack.setCurrentWidget(self.login_page)

    # =====================================================
    # OPEN MODULES (DEFERRED)
    # =====================================================
    def open_ocr(self):
        self.selection_page.set_cards_enabled(False)
        QTimer.singleShot(0, self._create_ocr)

    def open_live(self):
        self.selection_page.set_cards_enabled(False)
        QTimer.singleShot(0, self._create_live)

    def open_barcode(self):
        self.selection_page.set_cards_enabled(False)
        QTimer.singleShot(0, self._create_barcode)

    def open_barcode(self):
        self.selection_page.set_cards_enabled(False)
        QTimer.singleShot(0, self._create_barcode)

    def open_barcode_live(self):
        self.selection_page.set_cards_enabled(False)
        QTimer.singleShot(0, self._create_barcode_live)

    # =====================================================
    # CREATE MODULES (LAZY LOAD ONCE)
    # =====================================================
    def _create_ocr(self):
        if self.ocr_page is None:
            self.ocr_page = OCRGui()
            self.ocr_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.ocr_page)

        self.stack.setCurrentWidget(self.ocr_page)

    def _create_live(self):
        if self.live_page is None:
            self.live_page = OCRLiveGui()
            self.live_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.live_page)

        self.stack.setCurrentWidget(self.live_page)

    def _create_barcode(self):
        if self.barcode_page is None:
            self.barcode_page = BarcodeGui()
            self.barcode_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.barcode_page)

        self.stack.setCurrentWidget(self.barcode_page)

    def _create_barcode_live(self):
        if self.barcode_live_page is None:
            self.barcode_live_page = BarcodeLiveGui()
            self.barcode_live_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.barcode_live_page)

        self.stack.setCurrentWidget(self.barcode_live_page)


# =====================================================
# APPLICATION ENTRY
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
