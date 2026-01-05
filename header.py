import os
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, pyqtSignal


class Header(QWidget):
    logout_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # ================= HEADER SIZE =================
        self.setFixedHeight(150)

        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
            }
        """)

        # ================= LAYOUT =================
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(32, 15, 32, 15)
        self.layout.setSpacing(20)

        # ================= LOGO =================
        logo_path = os.path.join("assets", "logoo.svg")
        self.logo = QSvgWidget(logo_path)

        # Let layout & resizeEvent control size (IMPORTANT)
        self.logo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # ================= LOGOUT =================
        self.logout_btn = QPushButton("Logout")
        self.logout_btn.setFixedHeight(36)
        self.logout_btn.setCursor(Qt.PointingHandCursor)
        self.logout_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 18px;
                border-radius: 8px;
                border: 1px solid #d1d5db;
                background: #ffffff;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #f9fafb;
            }
        """)
        self.logout_btn.clicked.connect(self.logout_clicked.emit)

        # ================= ASSEMBLE =================
        self.layout.addWidget(self.logo, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.layout.addStretch()
        self.layout.addWidget(self.logout_btn)

    # =====================================================
    # RESPONSIVE LOGO SIZE (NO CLIPPING)
    # =====================================================
    def resizeEvent(self, event):
        """
        Ensures logo always fits inside header height.
        Width is scaled proportionally.
        """
        available_height = self.height() - (
            self.layout.contentsMargins().top() +
            self.layout.contentsMargins().bottom()
        )

        # Maintain wide-logo aspect ratio
        logo_height = max(60, available_height)
        logo_width = int(logo_height * 1.6)

        self.logo.setFixedSize(logo_width, logo_height)

        super().resizeEvent(event)

    # =====================================================
    # LOGO VISIBILITY CONTROL
    # =====================================================
    def set_logo_visible(self, visible: bool):
        if visible:
            self.logo.show()
        else:
            self.logo.hide()
