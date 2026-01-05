import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QStackedWidget
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer

from header import Header
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

        # ================= CENTRAL CONTAINER =================
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---------------- HEADER ----------------
        self.header = Header()
        layout.addWidget(self.header)

        # ---------------- STACK ----------------
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.setCentralWidget(container)

        # ================= PAGES =================
        self.login_page = LoginPage()
        self.selection_page = SelectionPage()

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.selection_page)

        # Lazy pages
        self.ocr_page = None
        self.live_page = None
        self.barcode_page = None
        self.barcode_live_page = None

        # ================= SIGNALS =================
        self.login_page.login_success.connect(self.show_selection)
        self.header.logout_clicked.connect(self.logout)

        self.selection_page.config_selected.connect(self.open_ocr)
        self.selection_page.live_selected.connect(self.open_live)
        self.selection_page.barcode_selected.connect(self.open_barcode)
        self.selection_page.barcode_live_selected.connect(self.open_barcode_live)

        # ================= INITIAL STATE =================
        self.header.show()
        self.header.set_logo_visible(True)
        self.stack.setCurrentWidget(self.login_page)

    # =====================================================
    # NAVIGATION
    # =====================================================
    def show_selection(self):
        self.header.show()
        self.header.set_logo_visible(True)
        self.selection_page.set_cards_enabled(True)
        self.stack.setCurrentWidget(self.selection_page)

    def logout(self):
        self.header.show()
        self.header.set_logo_visible(True)

        self.login_page.username_input.clear()
        self.login_page.password_input.clear()
        self.login_page.error_label.clear()

        self.stack.setCurrentWidget(self.login_page)

    # =====================================================
    # OPEN MODULES
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

    def open_barcode_live(self):
        self.selection_page.set_cards_enabled(False)
        QTimer.singleShot(0, self._create_barcode_live)

    # =====================================================
    # LAZY PAGE CREATION
    # =====================================================
    def _create_ocr(self):
        if self.ocr_page is None:
            self.ocr_page = OCRGui()
            self.ocr_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.ocr_page)

        self.header.hide()
        self.stack.setCurrentWidget(self.ocr_page)

    def _create_live(self):
        if self.live_page is None:
            self.live_page = OCRLiveGui()
            self.live_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.live_page)

        self.header.hide()
        self.stack.setCurrentWidget(self.live_page)

    def _create_barcode(self):
        if self.barcode_page is None:
            self.barcode_page = BarcodeGui()
            self.barcode_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.barcode_page)

        self.header.hide()
        self.stack.setCurrentWidget(self.barcode_page)

    def _create_barcode_live(self):
        if self.barcode_live_page is None:
            self.barcode_live_page = BarcodeLiveGui()
            self.barcode_live_page.back_to_selection.connect(self.show_selection)
            self.stack.addWidget(self.barcode_live_page)

        self.header.hide()
        self.stack.setCurrentWidget(self.barcode_live_page)


# ================= ENTRY POINT =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
