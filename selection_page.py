from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal


# ================================= ====================
# CARD WIDGET
# =====================================================
class SelectionCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, title: str, description: str):
        super().__init__()

        self.setFixedSize(360, 220)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("SelectionCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 26, 26, 26)
        layout.setSpacing(12)

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
    barcode_live_selected = pyqtSignal()

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
            background: #f9fafb;
        }

        QFrame#SelectionCard:disabled {
            background: #f1f5f9;
        }

        QLabel#CardTitle {
            color: #111827;
            font-size: 17px;
            font-weight: 600;
        }

        QLabel#CardDesc {
            color: #6b7280;
            font-size: 13px;
        }
        """)

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 40, 0, 40)
        center_layout.setSpacing(24)
        center_layout.setAlignment(Qt.AlignTop)

        header = QLabel("Computer Vision System")
        header.setObjectName("Header")
        header.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Choose your workflow")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        center_layout.addWidget(header)
        center_layout.addWidget(subtitle)

        cards_container = QWidget()
        cards_layout = QHBoxLayout(cards_container)
        cards_layout.setSpacing(28)
        cards_layout.setAlignment(Qt.AlignCenter)

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
        self.barcode_live_card = SelectionCard(
            "Barcode Live Feed",
            "Run real-time barcode detection using live camera streams with instant previews."
        )

        self.config_card.clicked.connect(self.config_selected.emit)
        self.live_card.clicked.connect(self.live_selected.emit)
        self.barcode_card.clicked.connect(self.barcode_selected.emit)
        self.barcode_live_card.clicked.connect(self.barcode_live_selected.emit)

        for card in (
            self.config_card,
            self.live_card,
            self.barcode_card,
            self.barcode_live_card
        ):
            cards_layout.addWidget(card)

        center_layout.addWidget(cards_container)

        footer = QLabel("Select a workflow to continue")
        footer.setObjectName("Footer")
        footer.setAlignment(Qt.AlignCenter)

        center_layout.addSpacing(30)
        center_layout.addWidget(footer)

        main_layout.addStretch()
        main_layout.addWidget(center)
        main_layout.addStretch()

    def set_cards_enabled(self, enabled: bool):
        for card in (
            self.config_card,
            self.live_card,
            self.barcode_card,
            self.barcode_live_card
        ):
            card.setEnabled(enabled)
