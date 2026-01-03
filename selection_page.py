from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal


# =====================================================
# CARD WIDGET
# =====================================================
class SelectionCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, title: str, description: str):
        super().__init__()

        self.setFixedSize(420, 260)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("SelectionCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setObjectName("CardDesc")

        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()

    def mousePressEvent(self, event):
        if self.isEnabled():
            self.clicked.emit()


# =====================================================
# SELECTION PAGE
# =====================================================
class SelectionPage(QWidget):
    config_selected = pyqtSignal()
    live_selected = pyqtSignal()
    barcode_selected = pyqtSignal()
    logout_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._apply_styles()
        self._build_ui()

    # -------------------------------------------------
    # STYLES
    # -------------------------------------------------
    def _apply_styles(self):
        self.setStyleSheet("""
        SelectionPage {
            background-color: #f5f6f8;
            font-family: 'Segoe UI';
        }

        QLabel#Header {
            color: #111827;
            font-size: 28px;
            font-weight: 600;
        }

        QLabel#Subtitle {
            color: #6b7280;
            font-size: 14px;
        }

        QLabel#Footer {
            color: #9ca3af;
            font-size: 12px;
        }

        QFrame#SelectionCard {
            background: white;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
        }

        QFrame#SelectionCard:hover {
            border: 1px solid #2563eb;
        }

        QFrame#SelectionCard:disabled {
            background: #f1f5f9;
            border: 1px solid #e5e7eb;
        }

        QLabel#CardTitle {
            color: #111827;
            font-size: 18px;
            font-weight: 600;
        }

        QLabel#CardDesc {
            color: #6b7280;
            font-size: 13px;
        }

        QPushButton#LogoutButton {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 13px;
            color: #374151;
        }

        QPushButton#LogoutButton:hover {
            background: #f9fafb;
        }
        """)

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(24)

        # ---------- TOP BAR ----------
        top_bar = QHBoxLayout()
        top_bar.addStretch()

        logout_btn = QPushButton("Logout")
        logout_btn.setObjectName("LogoutButton")
        logout_btn.clicked.connect(self.logout_clicked.emit)

        top_bar.addWidget(logout_btn)

        # ---------- HEADER ----------
        header = QLabel("OCR System")
        header.setObjectName("Header")
        header.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Choose your workflow")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        # ---------- CARDS ----------
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(40)
        cards_layout.setAlignment(Qt.AlignCenter)

        # Store references (IMPORTANT)
        self.config_card = SelectionCard(
            "OCR Configuration",
            "Configure preprocessing, test images, and run batch OCR with full control."
        )

        self.live_card = SelectionCard(
            "OCR Live Feed",
            "Run real-time OCR using live camera streams with instant previews."
        )

        self.barcode_card = SelectionCard(
            "Barcode Configuration",
            "Configure barcode rules, test images, and run barcode validation."
        )

        # Signal connections
        self.config_card.clicked.connect(self.config_selected.emit)
        self.live_card.clicked.connect(self.live_selected.emit)
        self.barcode_card.clicked.connect(self.barcode_selected.emit)

        cards_layout.addWidget(self.config_card)
        cards_layout.addWidget(self.live_card)
        cards_layout.addWidget(self.barcode_card)

        # ---------- FOOTER ----------
        footer = QLabel("Select a workflow to continue")
        footer.setObjectName("Footer")
        footer.setAlignment(Qt.AlignCenter)

        # ---------- ASSEMBLE ----------
        main_layout.addLayout(top_bar)
        main_layout.addSpacing(10)
        main_layout.addWidget(header)
        main_layout.addWidget(subtitle)
        main_layout.addStretch()
        main_layout.addLayout(cards_layout)
        main_layout.addStretch()
        main_layout.addWidget(footer)

    # -------------------------------------------------
    # CARD CONTROL (USED BY MAINWINDOW)
    # -------------------------------------------------
    def set_cards_enabled(self, enabled: bool):
        """
        Enable / disable all workflow cards.
        Called from MainWindow to avoid memory contention.
        """
        self.config_card.setEnabled(enabled)
        self.live_card.setEnabled(enabled)
        self.barcode_card.setEnabled(enabled)
