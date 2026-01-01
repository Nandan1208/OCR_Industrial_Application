import os, json, cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QTextEdit,
    QLineEdit, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

from ocr_engine import DoctrEngine, EasyOCREngine, PPOCREngine


class OCRGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Evaluation Tool")
        self.showMaximized()
        self.engine = DoctrEngine()
        self.BATCH_SIZE = 4

        self.original_image = None
        self.single_image = None

        self.folder_images = []
        self.batch_index = 0
        self.batch_running = False
        self.batch_paused = False

        self.batch_timer = QTimer()
        self.batch_timer.timeout.connect(self.run_batch_step)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        self.image_label = QLabel("Upload Image or Folder")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet(
            "border:1px solid #ccc; background:#fafafa"
        )

        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_folder_btn = QPushButton("Upload Folder")
        self.run_single_btn = QPushButton("Run OCR (Single)")
        self.run_batch_btn = QPushButton("Run Batch")

        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.enable_pre = QCheckBox("Enable Preprocessing")
        self.enable_pre.setChecked(True)
        self.enable_pre.stateChanged.connect(self.update_preview)

        self.use_clahe = QCheckBox("Enable CLAHE")
        self.use_clahe.stateChanged.connect(self.update_preview)

        self.brightness = self.make_slider("Brightness", -100, 100, 0)
        self.contrast = self.make_slider("Contrast", 10, 300, 100)
        self.gamma = self.make_slider("Gamma", 10, 300, 100)
        self.rotate = self.make_slider("Fine Rotate", -90, 90, 0)

        self.rotate_preset = QComboBox()
        self.rotate_preset.addItems(["0", "45", "90", "180", "270", "360"])
        self.rotate_preset.currentIndexChanged.connect(self.update_preview)

        self.ocr_selector = QComboBox()
        self.ocr_selector.addItems(["Doctr", "EasyOCR", "PaddleOCR"])

        self.regex = QLineEdit()
        self.regex.setPlaceholderText("Leave empty for RAW OCR")

        self.save_cfg_btn = QPushButton("💾 Save Preprocess Settings")
        self.load_cfg_btn = QPushButton("📂 Load Preprocess Settings")

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        left = QVBoxLayout()
        left.addWidget(self.image_label)

        right = QVBoxLayout()
        right.addWidget(QLabel("OCR Engine"))
        right.addWidget(self.ocr_selector)
        right.addWidget(self.upload_image_btn)
        right.addWidget(self.upload_folder_btn)
        right.addWidget(self.enable_pre)
        right.addWidget(self.use_clahe)
        right.addWidget(self.brightness[0])
        right.addWidget(self.contrast[0])
        right.addWidget(self.gamma[0])
        right.addWidget(self.rotate[0])
        right.addWidget(self.rotate_preset)
        right.addWidget(self.save_cfg_btn)
        right.addWidget(self.load_cfg_btn)
        right.addWidget(QLabel("Regex"))
        right.addWidget(self.regex)
        right.addWidget(self.run_single_btn)
        right.addWidget(self.run_batch_btn)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.resume_btn)
        btn_row.addWidget(self.stop_btn)
        right.addLayout(btn_row)

        right.addWidget(QLabel("OCR Output"))
        right.addWidget(self.output)
        right.addStretch()

        main = QHBoxLayout(self)
        main.addLayout(left, 3)
        main.addLayout(right, 1)

    # ==================================================
    # SLIDER
    # ==================================================
    def make_slider(self, name, mn, mx, val):
        label = QLabel(f"{name}: {val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)
        slider.valueChanged.connect(
            lambda v: (label.setText(f"{name}: {v}"), self.update_preview())
        )
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(label)
        layout.addWidget(slider)
        return box, slider

    # ==================================================
    # SIGNALS
    # ==================================================
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

    # ==================================================
    # ENGINE SWITCH
    # ==================================================
    def switch_engine(self):
        self.engine = {
            "Doctr": DoctrEngine,
            "EasyOCR": EasyOCREngine,
            "PaddleOCR": PPOCREngine
        }[self.ocr_selector.currentText()]()
        self.output.append("🔄 OCR engine switched")
        self.update_preview()

    # ==================================================
    # LOAD IMAGE / FOLDER
    # ==================================================
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
                if f.lower().endswith((".jpg", ".png", ".bmp", ".tif", ".tiff"))
            ])
            self.output.append(f"📂 Loaded {len(self.folder_images)} images")

    # ==================================================
    # PREVIEW
    # ==================================================
    def update_preview(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        if self.enable_pre.isChecked():
            total_rot = int(self.rotate_preset.currentText()) + self.rotate[1].value()
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

    # ==================================================
    # SINGLE OCR
    # ==================================================
    def run_single(self):
        if self.single_image is None:
            return

        # ✅ Clear console on every trigger
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

        # Show output
        if regex:
            self.output.append("\n".join(matches))
        else:
            self.output.append("\n".join(raw_text))

        # ✅ Save to JSON
        self.save_output_json(
            mode="single",
            results=[{
                "image": "single_image",
                "raw_text": raw_text,
                "matches": matches
            }]
        )

    # ==================================================
    # BATCH CONTROL
    # ==================================================
    def start_batch(self):
        if not self.folder_images:
            self.output.append("❌ No folder loaded")
            return

        self.batch_index = 0
        self.batch_running = True
        self.batch_paused = False
        self.batch_json_results = []  # ✅ reset

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
        self.output.append("⛔ Batch stopped")

    # ==================================================
    # BATCH STEP
    # ==================================================
    def run_batch_step(self):
        if not self.batch_running or self.batch_paused:
            return

        # -------- Batch completed --------
        if self.batch_index >= len(self.folder_images):
            self.batch_timer.stop()
            self.batch_running = False
            self.run_batch_btn.setEnabled(True)

            # ✅ Save batch output JSON once at end
            self.save_output_json(
                mode="batch",
                results=self.batch_json_results
            )

            self.output.clear()
            self.output.append("✅ Batch completed & output saved")
            return

        # -------- Process next batch --------
        paths = self.folder_images[self.batch_index:self.batch_index + self.BATCH_SIZE]
        images = []

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue

            # ✅ Apply CURRENT preprocessing UI values
            if self.enable_pre.isChecked():
                total_rot = int(self.rotate_preset.currentText()) + self.rotate[1].value()
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

        # -------- Per-image processing --------
        for p, img, res in zip(paths, images, results):
            # ✅ Clear output console per trigger
            self.output.clear()

            raw_text = self.engine.extract_all_text(res)
            matches = []

            if regex:
                matches = self.engine.extract_matches(res, regex)
                self.output.append(
                    f"{os.path.basename(p)} → {'MATCH' if matches else 'NO MATCH'}"
                )
            else:
                self.output.append(f"{os.path.basename(p)} → RAW OCR")

            # Store for JSON
            self.batch_json_results.append({
                "image": os.path.basename(p),
                "raw_text": raw_text,
                "matches": matches
            })

            # Show last processed image
            self.show_image(img)

        self.batch_index += self.BATCH_SIZE


    # ==================================================
    # SAVE / LOAD CONFIG
    # ==================================================
    def save_preprocess_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preprocess Config", "preprocess_config.json", "JSON (*.json)"
        )
        if not path:
            return

        config = {
            "ocr_model": self.ocr_selector.currentText(),
            "enable_preprocessing": self.enable_pre.isChecked(),
            "use_clahe": self.use_clahe.isChecked(),
            "brightness": self.brightness[1].value(),
            "contrast": self.contrast[1].value() / 100.0,
            "gamma": self.gamma[1].value() / 100.0,
            "fine_rotate": self.rotate[1].value(),
            "rotate_preset": int(self.rotate_preset.currentText())
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=4)

        self.output.append(f"💾 Saved config: {os.path.basename(path)}")

    def load_preprocess_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocess Config", "", "JSON (*.json)"
        )
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

        preset = str(config.get("rotate_preset", 0))
        idx = self.rotate_preset.findText(preset)
        if idx >= 0:
            self.rotate_preset.setCurrentIndex(idx)

        self.update_preview()
        self.output.append(f"📂 Loaded config: {os.path.basename(path)}")

    # ==================================================
    # DISPLAY
    # ==================================================
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
        """
        Save OCR results to JSON file.

        mode    : "single" or "batch"
        results : list of dicts with keys:
                - image
                - raw_text
                - matches
        """

        # Create output directory if not exists
        os.makedirs("ocr_outputs", exist_ok=True)

        payload = {
            "mode": mode,
            "engine": self.ocr_selector.currentText(),
            "regex": self.regex.text().strip(),
            "results": results
        }

        # Filename based on mode
        filename = f"ocr_output_{mode}.json"
        path = os.path.join("ocr_outputs", filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)

            # Log to console
            self.output.append(f"💾 Output saved: {path}")

        except Exception as e:
            self.output.append(f"❌ Failed to save output JSON: {e}")
