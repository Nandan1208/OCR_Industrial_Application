import os
import cv2
import json

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider,
    QTextEdit, QLineEdit, QCheckBox,
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
            color: #111827;
        }

        QPushButton {
            background: #0b2c6b;
            color: white;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: 600;
        }

        QPushButton:hover {
            background: #093070;
        }

        QGroupBox {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 14px;
            font-weight: 600;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #374151;
        }

        QLineEdit, QTextEdit {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 6px;
            color: #111827;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #e5e7eb;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
            background: #2563eb;
        }
        """)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ===== LEFT PANEL =====
        left_panel = QVBoxLayout()

        back_btn = QPushButton("Back")
        back_btn.setFixedWidth(120)
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.image_label = QLabel("Upload image or folder")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(720, 520)
        self.image_label.setStyleSheet("""
            background: white;
            border: 2px dashed #c7d2fe;
            border-radius: 14px;
            color: #64748b;
            font-size: 14px;
        """)

        left_panel.addWidget(back_btn, alignment=Qt.AlignLeft)
        left_panel.addWidget(self.image_label)

        # ===== RIGHT PANEL =====
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(420)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(14)

        # Upload
        upload_group = QGroupBox("Upload")
        ul = QVBoxLayout()
        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_folder_btn = QPushButton("Upload Folder")
        ul.addWidget(self.upload_image_btn)
        ul.addWidget(self.upload_folder_btn)
        upload_group.setLayout(ul)

        # Preprocessing
        preprocess_group = QGroupBox("Preprocessing")
        pl = QVBoxLayout()
        self.enable_pre = QCheckBox("Enable preprocessing")
        self.enable_pre.setChecked(True)
        self.enable_pre.stateChanged.connect(self.update_preview)

        self.use_clahe = QCheckBox("Enable CLAHE")
        self.use_clahe.stateChanged.connect(self.update_preview)
        self.brightness = self._make_slider("Brightness", -100, 100, 0)
        self.contrast = self._make_slider("Contrast", 10, 300, 100)
        self.gamma = self._make_slider("Gamma", 10, 300, 100)
        self.rotate = self._make_slider("Rotate", -90, 90, 0)
        

        pl.addWidget(self.enable_pre)
        pl.addWidget(self.use_clahe)
        pl.addWidget(self.brightness[0])
        pl.addWidget(self.contrast[0])
        pl.addWidget(self.gamma[0])
        pl.addWidget(self.rotate[0])
        preprocess_group.setLayout(pl)

        # Match input
        match_group = QGroupBox("Barcode Validation")
        ml = QVBoxLayout()
        self.expected_input = QLineEdit()
        self.expected_input.setPlaceholderText("Expected barcode value")
        ml.addWidget(QLabel("Expected value"))
        ml.addWidget(self.expected_input)
        match_group.setLayout(ml)

        # Actions
        batch_controls = QHBoxLayout()
        self.run_single_btn = QPushButton("Run (Single)")
        self.run_batch_btn = QPushButton("Run (Folder)")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        batch_controls.addWidget(self.pause_btn)
        batch_controls.addWidget(self.resume_btn)
        batch_controls.addWidget(self.stop_btn)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFixedHeight(160)

        right.addWidget(upload_group)
        right.addWidget(preprocess_group)
        right.addWidget(match_group)
        right.addWidget(self.run_single_btn)
        right.addWidget(self.run_batch_btn)
        right.addLayout(batch_controls)
        right.addWidget(QLabel("Output"))
        right.addWidget(self.output)
        right.addStretch()

        right_scroll.setWidget(right_widget)

        root.addLayout(left_panel, 3)
        root.addWidget(right_scroll, 1)

    # ---------------- HELPERS ----------------
    def _make_slider(self, name, mn, mx, val):
        label = QLabel(f"{name}: {val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)
        slider.valueChanged.connect(
            lambda v: label.setText(f"{name}: {v}")
        )

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

        # Live preview
        for _, s in [self.brightness, self.contrast, self.gamma, self.rotate]:
            s.valueChanged.connect(self.update_preview)
        self.enable_pre.stateChanged.connect(self.update_preview)

    # ---------------- LOGIC ----------------
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
                if f.lower().endswith((".png", ".jpg", ".bmp", ".tiff", ".jpeg"))
            ]
            self.output.append(f"Loaded {len(self.folder_images)} images")
    
    def update_preview(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        if self.enable_pre.isChecked():
            
            img = self.engine.preprocess(
                img,
                self.brightness[1].value(),
                self.contrast[1].value() / 100.0,
                self.gamma[1].value() / 100.0,
                
                self.use_clahe.isChecked()
            )
        self.single_image = img
        self.show_image(img)

    def preprocess(self, img):
        if not self.enable_pre.isChecked():
            return img

        return self.engine.preprocess(
            img,
            self.brightness[1].value(),
            self.contrast[1].value() / 100.0,
            self.gamma[1].value() / 100.0,
            self.rotate[1].value(),
            False
        )

    # ---------------- SINGLE ----------------
    def run_single(self):
        if self.original_image is None:
            return

        display_img = self.preprocess(self.original_image.copy())
        result = self.engine.run_batch([display_img])[0]

        vis = self.engine.draw_matches(
            display_img,
            result,
            self.expected_input.text().strip()
        )

        self.show_image(vis)

        matches, values = self.engine.extract_matches(
            result,
            self.expected_input.text().strip()
        )

        self.output.clear()
        self.output.append(f"Detected: {values}")
        self.output.append("MATCH" if matches else "NOT MATCH")

    # ---------------- BATCH ----------------
    def run_batch(self):
        if not self.folder_images:
            return

        self.batch_index = 0
        self.batch_results = []
        self.output.clear()
        self.output.append("🚀 Batch started")

        self.batch_timer = QTimer(self)
        self.batch_timer.timeout.connect(self.run_batch_step)
        self.batch_timer.start(500)

    def run_batch_step(self):
        if self.batch_index >= len(self.folder_images):
            self.batch_timer.stop()

            with open("barcode_results.json", "w", encoding="utf-8") as f:
                json.dump(self.batch_results, f, indent=4)

            self.output.append("✅ Batch completed")
            return

        path = self.folder_images[self.batch_index]
        img = cv2.imread(path)

        if img is None:
            self.batch_index += 1
            return

        display_img = self.preprocess(img.copy())
        result = self.engine.run_batch([display_img])[0]

        vis = self.engine.draw_matches(
            display_img,
            result,
            self.expected_input.text().strip()
        )

        self.show_image(vis)

        matches, values = self.engine.extract_matches(
            result,
            self.expected_input.text().strip()
        )

        status = "MATCH" if matches else "NOT MATCH"
        self.output.append(f"{os.path.basename(path)} → {status}")

        self.batch_results.append({
            "image": os.path.basename(path),
            "values": values,
            "match": bool(matches)
        })

        self.batch_index += 1

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
    def pause_batch(self):
        self.batch_paused = True
        self.batch_timer.stop()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)

    def resume_batch(self):
        self.batch_paused = False
        self.batch_timer.start(250)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)

    def stop_batch(self):
        self.batch_timer.stop()
        self.batch_running = False
        self.run_batch_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.output.append("⛔ Stopped")