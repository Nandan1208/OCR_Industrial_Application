from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QLineEdit, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class LoginPage(QWidget):
    login_success = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.valid_username = "admin"
        self.valid_password = "admin123"

        self._apply_styles()
        self._build_ui()

    # ---------------- STYLES ----------------
    def _apply_styles(self):
        self.setStyleSheet("""
        /* Root background */
        LoginPage {
            background-color: #f5f6f8;
            font-family: 'Segoe UI';
        }

        /* Login Card */
        QFrame#LoginCard {
            background: white;
            border-radius: 14px;
        }

        QLabel#Logo {
            color: #9aa4b2;
            font-size: 13px;
            font-weight: 600;
        }

        QLabel#Title {
            color: #111827;
            font-size: 22px;
            font-weight: 600;
        }

        QLabel#Subtitle {
            color: #6b7280;
            font-size: 13px;
        }

        QLabel#Footer {
            color: #9ca3af;
            font-size: 12px;
        }

        QLabel#Error {
            font-size: 12px;
            font-weight: 600;
        }

        QLineEdit {
            background: #f9fafb;
            color: #111827;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
        }

        QLineEdit:focus {
            border: 1px solid #2563eb;
            background: white;
        }

        QPushButton#LoginButton {
            background-color: #0b2c6b;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            font-weight: 600;
        }

        QPushButton#LoginButton:hover {
            background-color: #093070;
        }

        QPushButton#LoginButton:pressed {
            background-color: #07265a;
        }
        """)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("LoginCard")
        card.setFixedWidth(420)

        # subtle shadow using margin illusion
        card.setStyleSheet("""
            QFrame#LoginCard {
                margin: 10px;
            }
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(18)

        # logo = QLabel("Your logo")
        # logo.setObjectName("Logo")
        # logo.setAlignment(Qt.AlignCenter)

        title = QLabel("Login")
        title.setObjectName("Title")

        subtitle = QLabel("Sign in to continue")
        subtitle.setObjectName("Subtitle")

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Email")
        self.username_input.returnPressed.connect(self.handle_login)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.handle_login)

        self.error_label = QLabel("")
        self.error_label.setObjectName("Error")
        self.error_label.setAlignment(Qt.AlignCenter)

        login_btn = QPushButton("Sign in")
        login_btn.setObjectName("LoginButton")
        login_btn.clicked.connect(self.handle_login)
        login_btn.setCursor(Qt.PointingHandCursor)

        footer = QLabel("Don’t have an account? Register for free")
        footer.setObjectName("Footer")
        footer.setAlignment(Qt.AlignCenter)

        # layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.error_label)
        layout.addSpacing(10)
        layout.addWidget(login_btn)
        layout.addSpacing(15)
        layout.addWidget(footer)

        root.addWidget(card)

        self.username_input.setFocus()

    # ---------------- LOGIC ----------------
    def handle_login(self):
        if (
            self.username_input.text().strip() == self.valid_username and
            self.password_input.text() == self.valid_password
        ):
            self.error_label.setStyleSheet("color:#16a34a;")
            self.error_label.setText("Login successful")
            self.login_success.emit()
        else:
            self.error_label.setStyleSheet("color:#dc2626;")
            self.error_label.setText("Invalid email or password")
            self.password_input.clear()
            self.password_input.setFocus()
