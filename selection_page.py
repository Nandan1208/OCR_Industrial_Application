from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


# ---------------- CARD ----------------
class SelectionCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, title, description):
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
        self.clicked.emit()


# ---------------- PAGE ----------------
class SelectionPage(QWidget):
    config_selected = pyqtSignal()
    live_selected = pyqtSignal()
    logout_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._apply_styles()
        self._build_ui()

    # ---------------- STYLES ----------------
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

    # ---------------- UI ----------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(24)

        # Header
        top_bar = QHBoxLayout()
        top_bar.addStretch()

        logout_btn = QPushButton("Logout")
        logout_btn.setObjectName("LogoutButton")
        logout_btn.clicked.connect(self.logout_clicked.emit)
        logout_btn.setCursor(Qt.PointingHandCursor)

        top_bar.addWidget(logout_btn)

        header = QLabel("OCR System")
        header.setObjectName("Header")
        header.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Choose your workflow")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        # Cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(40)
        cards_layout.setAlignment(Qt.AlignCenter)

        config_card = SelectionCard(
            "OCR Configuration",
            "Configure preprocessing, test images, and run batch OCR with full control."
        )
        config_card.clicked.connect(self.config_selected.emit)

        live_card = SelectionCard(
            "OCR Live Feed",
            "Run real-time OCR using live camera streams with instant previews."
        )
        live_card.clicked.connect(self.live_selected.emit)

        cards_layout.addWidget(config_card)
        cards_layout.addWidget(live_card)

        footer = QLabel("Select a workflow to continue")
        footer.setObjectName("Footer")
        footer.setAlignment(Qt.AlignCenter)

        main_layout.addLayout(top_bar)
        main_layout.addSpacing(10)
        main_layout.addWidget(header)
        main_layout.addWidget(subtitle)
        main_layout.addStretch()
        main_layout.addLayout(cards_layout)
        main_layout.addStretch()
        main_layout.addWidget(footer)
