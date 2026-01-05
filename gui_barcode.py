import os
import cv2
import json

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider,
    QTextEdit, QLineEdit, QCheckBox, QComboBox,
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

from ocr_engine import BarcodeEngine


class BarcodeGui(QWidget):
    back_to_selection = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.engine = BarcodeEngine()

        self.original_image = None
        self.single_image = None
        self.folder_images = []

        # Batch state
        self.batch_index = 0
        self.batch_running = False
        self.batch_paused = False
        self.batch_results = []
        self.batch_timer = None

        self._apply_styles()
        self._build_ui()
        self._connect_signals()

    # ---------------- STYLES ----------------
    def _apply_styles(self):
        self.setStyleSheet("""
        BarcodeGui {
            background-color: #f5f6f8;
            font-family: 'Segoe UI';
            font-size: 10pt;
        }
        QPushButton {
            background: #0b2c6b;
            color: white;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QPushButton:hover { background: #093070; }
        QGroupBox {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 14px;
        }
        QLineEdit, QTextEdit {
            background: #f9fafb;
            border-radius: 6px;
            padding: 6px;
        }
        """)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # ===== LEFT =====
        left = QVBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.image_label = QLabel("Upload image or folder")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(720, 520)
        self.image_label.setStyleSheet("border:2px dashed #c7d2fe;")

        left.addWidget(back_btn, alignment=Qt.AlignLeft)
        left.addWidget(self.image_label)

        # ===== RIGHT =====
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(420)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)

        # Upload
        upload_group = QGroupBox("Upload")
        ul = QVBoxLayout(upload_group)
        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_folder_btn = QPushButton("Upload Folder")
        ul.addWidget(self.upload_image_btn)
        ul.addWidget(self.upload_folder_btn)

        # Preprocessing
        preprocess_group = QGroupBox("Preprocessing")
        pl = QVBoxLayout(preprocess_group)

        self.enable_pre = QCheckBox("Enable preprocessing")
        self.enable_pre.setChecked(True)

        self.use_clahe = QCheckBox("Enable CLAHE")

        self.brightness = self._make_slider("Brightness", -100, 100, 0)
        self.contrast = self._make_slider("Contrast", 10, 300, 100)
        self.gamma = self._make_slider("Gamma", 10, 300, 100)
        self.rotate = self._make_slider("Rotate fine", -90, 90, 0)

        self.rotate_preset = QComboBox()
        self.rotate_preset.addItems(["0°", "90°", "180°", "270°"])

        pl.addWidget(self.enable_pre)
        pl.addWidget(self.use_clahe)
        pl.addWidget(self.brightness[0])
        pl.addWidget(self.contrast[0])
        pl.addWidget(self.gamma[0])
        pl.addWidget(self.rotate[0])
        pl.addWidget(QLabel("Rotation preset"))
        pl.addWidget(self.rotate_preset)

        # Match
        match_group = QGroupBox("Barcode Validation")
        ml = QVBoxLayout(match_group)
        self.expected_input = QLineEdit()
        self.expected_input.setPlaceholderText("Expected barcode value")
        ml.addWidget(self.expected_input)

        # Controls
        self.run_single_btn = QPushButton("Run (Single)")
        self.run_batch_btn = QPushButton("Run (Folder)")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFixedHeight(160)

        right.addWidget(upload_group)
        right.addWidget(preprocess_group)
        right.addWidget(match_group)
        right.addWidget(self.run_single_btn)
        right.addWidget(self.run_batch_btn)
        right.addWidget(self.pause_btn)
        right.addWidget(self.resume_btn)
        right.addWidget(self.stop_btn)
        right.addWidget(self.output)

        right_scroll.setWidget(right_widget)

        root.addLayout(left, 3)
        root.addWidget(right_scroll, 1)

    # ---------------- HELPERS ----------------
    def _make_slider(self, name, mn, mx, val):
        label = QLabel(f"{name}: {val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)
        slider.valueChanged.connect(lambda v: label.setText(f"{name}: {v}"))
        box = QWidget()
        l = QVBoxLayout(box)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(label)
        l.addWidget(slider)
        return box, slider

    # ---------------- SIGNALS ----------------
    def _connect_signals(self):
        self.upload_image_btn.clicked.connect(self.load_image)
        self.upload_folder_btn.clicked.connect(self.load_folder)
        self.run_single_btn.clicked.connect(self.run_single)
        self.run_batch_btn.clicked.connect(self.run_batch)
        self.pause_btn.clicked.connect(self.pause_batch)
        self.resume_btn.clicked.connect(self.resume_batch)
        self.stop_btn.clicked.connect(self.stop_batch)

        for _, s in [self.brightness, self.contrast, self.gamma, self.rotate]:
            s.valueChanged.connect(self.update_preview)

        self.enable_pre.stateChanged.connect(self.update_preview)
        self.rotate_preset.currentIndexChanged.connect(self.update_preview)

    # ---------------- LOGIC ----------------
    def get_rotation_angle(self):
        preset = int(self.rotate_preset.currentText().replace("°", ""))
        fine = self.rotate[1].value()
        return (preset + fine) % 360

    def preprocess(self, img):
        if not self.enable_pre.isChecked():
            return img

        return self.engine.preprocess(
            img,
            self.brightness[1].value(),
            self.contrast[1].value() / 100.0,
            self.gamma[1].value() / 100.0,
            self.get_rotation_angle(),
            self.use_clahe.isChecked()
        )

    def update_preview(self):
        if self.original_image is None:
            return
        img = self.preprocess(self.original_image.copy())
        self.show_image(img)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image")
        if not path:
            return
        self.original_image = cv2.imread(path)
        self.show_image(self.original_image)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_images = [
                os.path.join(folder, f)
                for f in sorted(os.listdir(folder))
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
            self.output.append(f"Loaded {len(self.folder_images)} images")

    # ---------------- SINGLE ----------------
    def run_single(self):
        if self.original_image is None:
            return

        img = self.preprocess(self.original_image.copy())
        result = self.engine.run_batch([img])[0]

        matches, values = self.engine.extract_matches(
            result,
            self.expected_input.text().strip()
        )

        if not values:
            status = "NO BARCODE"
        elif matches:
            status = "MATCH"
        else:
            status = "NOT MATCH"

        img = self.draw_status_text(img, status)

        self.output.setText(f"Detected: {values}\n{status}")
        self.show_image(img)


    # ---------------- BATCH ----------------
    def run_batch(self):
        if not self.folder_images:
            return

        self.batch_index = 0
        self.batch_results.clear()
        self.batch_running = True

        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self.batch_timer = QTimer(self)
        self.batch_timer.timeout.connect(self.run_batch_step)
        self.batch_timer.start(400)

    def run_batch_step(self):
        if self.batch_index >= len(self.folder_images):
            self.batch_timer.stop()
            self.batch_running = False

            with open("barcode_results.json", "w", encoding="utf-8") as f:
                json.dump(self.batch_results, f, indent=4)

            self.output.append(" Batch completed")
            return

        path = self.folder_images[self.batch_index]
        img = cv2.imread(path)

        if img is None:
            self.batch_index += 1
            return

        img = self.preprocess(img)
        result = self.engine.run_batch([img])[0]

        matches, values = self.engine.extract_matches(
            result,
            self.expected_input.text().strip()
        )

        if not values:
            status = "NO BARCODE"
        elif matches:
            status = "MATCH"
        else:
            status = "NOT MATCH"

        img = self.draw_status_text(img, status)

        self.output.append(f"{os.path.basename(path)} → {status}")

        self.batch_results.append({
            "image": os.path.basename(path),
            "values": values,
            "status": status
        })

        self.show_image(img)
        self.batch_index += 1


    # ---------------- CONTROLS ----------------
    def pause_batch(self):
        if self.batch_timer:
            self.batch_timer.stop()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_batch(self):
        if self.batch_timer:
            self.batch_timer.start(400)
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def stop_batch(self):
        if self.batch_timer:
            self.batch_timer.stop()
        self.batch_running = False
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.output.append("⛔ Batch stopped")
    def draw_status_text(self, img, status):
        color_map = {
            "MATCH": (0, 255, 0),
            "NOT MATCH": (0, 165, 255),
            "NO BARCODE": (0, 0, 255)
        }

        cv2.putText(
            img,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color_map.get(status, (255, 255, 255)),
            3,
            cv2.LINE_AA
        )
        return img
    # ---------------- DISPLAY ----------------
    def show_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)
