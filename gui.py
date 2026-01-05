import os, json, cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QTextEdit,
    QLineEdit, QCheckBox, QComboBox, QGroupBox,
    QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from ocr_engine import DoctrEngine, EasyOCREngine, PPOCREngine


class OCRGui(QWidget):
    back_to_selection = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.engine = EasyOCREngine()
        self.BATCH_SIZE = 4

        self.original_image = None
        self.single_image = None
        self.folder_images = []

        self.batch_index = 0
        self.batch_running = False
        self.batch_paused = False

        self.batch_timer = QTimer()
        self.batch_timer.timeout.connect(self.run_batch_step)

        self._apply_styles()
        self._build_ui()
        self._connect_signals()

    # ---------------- STYLES ----------------
    def _apply_styles(self):
        self.setStyleSheet("""
        OCRGui {
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

        QPushButton:disabled {
            background: #cbd5e1;
            color: #64748b;
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

        QLabel {
            color: #374151;
        }

        QLineEdit, QTextEdit, QComboBox {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 6px 8px;
            color: #111827;
        }

        QLineEdit:focus, QTextEdit:focus {
            border: 1px solid #2563eb;
            background: white;
        }

        QCheckBox {
            color: #374151;
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

        QScrollArea {
            border: none;
        }
        """)

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        # -------- LEFT PANEL (IMAGE) --------
        left = QVBoxLayout()

        back_btn = QPushButton("Back")
        back_btn.setMaximumWidth(120)
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.image_label = QLabel("Upload image or folder to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(720, 520)
        self.image_label.setStyleSheet("""
            background: white;
            border: 2px dashed #c7d2fe;
            border-radius: 14px;
            color: #64748b;
            font-size: 14px;
        """)

        left.addWidget(back_btn, alignment=Qt.AlignLeft)
        left.addWidget(self.image_label)

        # -------- RIGHT PANEL (CONTROLS) --------
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
        self.brightness = self.make_slider("Brightness", -100, 100, 0)
        self.contrast = self.make_slider("Contrast", 10, 300, 100)
        self.gamma = self.make_slider("Gamma", 10, 300, 100)
        self.rotate = self.make_slider("Rotate", -90, 90, 0)

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
        preprocess_group.setLayout(pl)

        # OCR settings
        ocr_group = QGroupBox("OCR Settings")
        ol = QVBoxLayout()
        self.ocr_selector = QComboBox()
        self.ocr_selector.addItems(["Model - 1", "Model - 2", "Model - 3"])
        self.regex = QLineEdit()
        self.regex.setPlaceholderText("Optional regex filter")
        ol.addWidget(QLabel("Select model"))
        ol.addWidget(self.ocr_selector)
        ol.addWidget(QLabel("Regex"))
        ol.addWidget(self.regex)
        ocr_group.setLayout(ol)

        # Actions
        self.save_cfg_btn = QPushButton("Save config")
        self.load_cfg_btn = QPushButton("Load config")
        self.run_single_btn = QPushButton("Run OCR (Single)")
        self.run_batch_btn = QPushButton("Run Batch")

        batch_controls = QHBoxLayout()
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
        self.output.setMaximumHeight(160)

        # Assemble
        right.addWidget(upload_group)
        right.addWidget(preprocess_group)
        right.addWidget(ocr_group)
        right.addWidget(self.save_cfg_btn)
        right.addWidget(self.load_cfg_btn)
        right.addWidget(self.run_single_btn)
        right.addWidget(self.run_batch_btn)
        right.addLayout(batch_controls)
        right.addWidget(QLabel("Output"))
        right.addWidget(self.output)
        right.addStretch()

        right_scroll.setWidget(right_widget)

        main.addLayout(left, 3)
        main.addWidget(right_scroll, 1)

    # ---------------- HELPERS ----------------
    def make_slider(self, name, mn, mx, val):
        label = QLabel(f"{name}: {val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)
        slider.valueChanged.connect(lambda v: (label.setText(f"{name}: {v}"), self.update_preview()))
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(slider)
        return box, slider

    def _connect_signals(self):
        self.upload_image_btn.clicked.connect(self.load_image)
        self.upload_folder_btn.clicked.connect(self.load_folder)
        self.run_single_btn.clicked.connect(self.run_single)
        self.run_batch_btn.clicked.connect(self.start_batch)
        self.pause_btn.clicked.connect(self.pause_batch)
        self.resume_btn.clicked.connect(self.resume_batch)
        self.stop_btn.clicked.connect(self.stop_batch)
        self.ocr_selector.currentIndexChanged.connect(self.switch_engine)
        self.save_cfg_btn.clicked.connect(self.save_preprocess_config)
        self.load_cfg_btn.clicked.connect(self.load_preprocess_config)

    def switch_engine(self):
        self.engine = {
            "Model - 1": DoctrEngine,
            "Model - 2": EasyOCREngine,
            "Model - 3": PPOCREngine
        }[self.ocr_selector.currentText()]()
        self.output.append("🔄 Engine switched")
        self.update_preview()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image")
        if path:
            self.original_image = cv2.imread(path)
            self.update_preview()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_images = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png", ".bmp",  ".tiff"))
            ])
            self.output.append(f" Loaded {len(self.folder_images)} images")

    def update_preview(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        if self.enable_pre.isChecked():
            preset_text = self.rotate_preset.currentText().replace("°", "")
            total_rot = int(preset_text) + self.rotate[1].value()
            img = self.engine.preprocess(
                img,
                self.brightness[1].value(),
                self.contrast[1].value() / 100.0,
                self.gamma[1].value() / 100.0,
                total_rot,
                self.use_clahe.isChecked()
            )
        self.single_image = img
        self.show_image(img)

    def run_single(self):
        if self.single_image is None:
            return
        self.output.clear()
        regex = self.regex.text().strip()
        img = self.single_image.copy()
        result = self.engine.run_batch([img])[0]
        raw_text = self.engine.extract_all_text(result)
        matches = []
        if regex:
            matches = self.engine.extract_matches(result, regex)
            img = self.engine.draw_matches(img, result, regex)
        self.show_image(img)
        if regex:
            self.output.append("\n".join(matches))
        else:
            self.output.append("\n".join(raw_text))
        self.save_output_json(
            mode="single",
            results=[{"image": "single_image", "raw_text": raw_text, "matches": matches}]
        )

    def start_batch(self):
        if not self.folder_images:
            self.output.append(" No folder loaded")
            return
        self.batch_index = 0
        self.batch_running = True
        self.batch_paused = False
        self.batch_json_results = []
        self.run_batch_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.output.clear()
        self.output.append("🚀 Batch started")
        self.batch_timer.start(250)

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

    def run_batch_step(self):
        if not self.batch_running or self.batch_paused:
            return
        if self.batch_index >= len(self.folder_images):
            self.batch_timer.stop()
            self.batch_running = False
            self.run_batch_btn.setEnabled(True)
            self.save_output_json(mode="batch", results=self.batch_json_results)
            self.output.clear()
            self.output.append("✅ Batch completed")
            return
        paths = self.folder_images[self.batch_index:self.batch_index + self.BATCH_SIZE]
        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            if self.enable_pre.isChecked():
                preset_text = self.rotate_preset.currentText().replace("°", "")
                total_rot = int(preset_text) + self.rotate[1].value()
                img = self.engine.preprocess(
                    img,
                    self.brightness[1].value(),
                    self.contrast[1].value() / 100.0,
                    self.gamma[1].value() / 100.0,
                    total_rot,
                    self.use_clahe.isChecked()
                )
            images.append(img)
        if not images:
            self.batch_index += self.BATCH_SIZE
            return
        results = self.engine.run_batch(images)
        regex = self.regex.text().strip()
        for p, img, res in zip(paths, images, results):
            self.output.clear()
            raw_text = self.engine.extract_all_text(res)
            matches = []
            if regex:
                matches = self.engine.extract_matches(res, regex)
                self.output.append(f"{os.path.basename(p)} → {'✓' if matches else '✗'}")
            else:
                self.output.append(f"{os.path.basename(p)} → RAW")
            self.batch_json_results.append({
                "image": os.path.basename(p),
                "raw_text": raw_text,
                "matches": matches
            })
            self.show_image(img)
        self.batch_index += self.BATCH_SIZE

    def save_preprocess_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "config.json", "JSON (*.json)"
        )
        if not path:
            return
        preset_text = self.rotate_preset.currentText().replace("°", "")
        config = {
            "ocr_model": self.ocr_selector.currentText(),
            "enable_preprocessing": self.enable_pre.isChecked(),
            "use_clahe": self.use_clahe.isChecked(),
            "brightness": self.brightness[1].value(),
            "contrast": self.contrast[1].value() / 100.0,
            "gamma": self.gamma[1].value() / 100.0,
            "fine_rotate": self.rotate[1].value(),
            "rotate_preset": int(preset_text)
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        self.output.append(f"💾 Saved: {os.path.basename(path)}")

    def load_preprocess_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            config = json.load(f)
        model = config.get("ocr_model", "Doctr")
        idx = self.ocr_selector.findText(model)
        if idx >= 0:
            self.ocr_selector.setCurrentIndex(idx)
        self.enable_pre.setChecked(config.get("enable_preprocessing", True))
        self.use_clahe.setChecked(config.get("use_clahe", False))
        self.brightness[1].setValue(config.get("brightness", 0))
        self.contrast[1].setValue(int(config.get("contrast", 1.0) * 100))
        self.gamma[1].setValue(int(config.get("gamma", 1.0) * 100))
        self.rotate[1].setValue(config.get("fine_rotate", 0))
        preset = str(config.get("rotate_preset", 0)) + "°"
        idx = self.rotate_preset.findText(preset)
        if idx >= 0:
            self.rotate_preset.setCurrentIndex(idx)
        self.update_preview()
        self.output.append(f"📂 Loaded: {os.path.basename(path)}")

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

    def save_output_json(self, mode, results):
        os.makedirs("ocr_outputs", exist_ok=True)
        payload = {
            "mode": mode,
            "engine": self.ocr_selector.currentText(),
            "regex": self.regex.text().strip(),
            "results": results
        }
        filename = f"ocr_output_{mode}.json"
        path = os.path.join("ocr_outputs", filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            self.output.append(f" Saved: {path}")
        except Exception as e:
            self.output.append(f" Error: {e}")