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
import re

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
        QPushButton:hover { background: #093070; }
        QPushButton:disabled { background: #cbd5e1; color: #64748b; }
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
        QLineEdit, QTextEdit, QComboBox {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 6px 8px;
        }
        QTextEdit { font-family: Consolas, monospace; }
        """)

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QHBoxLayout(self)

        # LEFT (IMAGE)
        left = QVBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.image_label = QLabel("Upload image or folder")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(720, 520)
        self.image_label.setStyleSheet(
            "background:white; border:2px dashed #c7d2fe; border-radius:14px;"
        )

        left.addWidget(back_btn, alignment=Qt.AlignLeft)
        left.addWidget(self.image_label)

        # RIGHT (CONTROLS)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(440)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(12)

        # Upload
        upload_group = QGroupBox("Upload")
        ul = QVBoxLayout()
        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_folder_btn = QPushButton("Upload Folder")
        ul.addWidget(self.upload_image_btn)
        ul.addWidget(self.upload_folder_btn)
        upload_group.setLayout(ul)

        # Preprocess
        preprocess_group = QGroupBox("Preprocessing")
        pl = QVBoxLayout()
        self.enable_pre = QCheckBox("Enable preprocessing")
        self.enable_pre.setChecked(True)
        self.use_clahe = QCheckBox("Enable CLAHE")

        self.brightness = self.make_slider("Brightness", -100, 100, 0)
        self.contrast = self.make_slider("Contrast", 10, 300, 100)
        self.gamma = self.make_slider("Gamma", 10, 300, 100)
        self.rotate = self.make_slider("Rotate", -90, 90, 0)

        self.rotate_preset = QComboBox()
        self.rotate_preset.addItems(["0°", "90°", "180°", "270°"])

        for w in [
            self.enable_pre, self.use_clahe,
            self.brightness[0], self.contrast[0],
            self.gamma[0], self.rotate[0],
            QLabel("Rotation preset"), self.rotate_preset
        ]:
            pl.addWidget(w)
        preprocess_group.setLayout(pl)

        # OCR Settings
        ocr_group = QGroupBox("OCR Settings")
        ol = QVBoxLayout()
        self.ocr_selector = QComboBox()
        self.ocr_selector.addItems(["Model - 1", "Model - 2", "Model - 3"])
        self.regex = QLineEdit()
        self.regex.setPlaceholderText("Optional regex")
        self.char_count_input = QLineEdit()
        self.char_count_input.setPlaceholderText("Expected character count")

        for w in [
            QLabel("Model"), self.ocr_selector,
            QLabel("Regex"), self.regex,
            QLabel("Expected character count"), self.char_count_input
        ]:
            ol.addWidget(w)
        ocr_group.setLayout(ol)

        # Actions
        self.save_cfg_btn = QPushButton("Save config")
        self.load_cfg_btn = QPushButton("Load config")
        self.run_single_btn = QPushButton("Run OCR (Single)")
        self.run_batch_btn = QPushButton("Run Batch")

        batch_ctrl_widget = QWidget()
        batch_ctrl_layout = QHBoxLayout(batch_ctrl_widget)
        batch_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")

        for b in [self.pause_btn, self.resume_btn, self.stop_btn]:
            b.setEnabled(False)
            batch_ctrl_layout.addWidget(b)

        # OCR Output (SEPARATE BOX)
        ocr_out_group = QGroupBox("OCR Output")
        ool = QVBoxLayout()
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        ool.addWidget(self.output)
        ocr_out_group.setLayout(ool)

        for w in [
            upload_group, preprocess_group, ocr_group,
            self.save_cfg_btn, self.load_cfg_btn,
            self.run_single_btn, self.run_batch_btn,
            batch_ctrl_widget,     
            ocr_out_group
        ]:
            right.addWidget(w)


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
        slider.valueChanged.connect(
            lambda v: (label.setText(f"{name}: {v}"), self.update_preview())
        )
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(label)
        layout.addWidget(slider)
        return box, slider

    def _connect_signals(self):
        self.upload_image_btn.clicked.connect(self.load_image)
        self.upload_folder_btn.clicked.connect(self.load_folder)
        self.run_single_btn.clicked.connect(self.run_single)
        self.run_batch_btn.clicked.connect(self.start_batch)
        self.use_clahe.stateChanged.connect(self.update_preview)
        self.pause_btn.clicked.connect(self.pause_batch)
        self.resume_btn.clicked.connect(self.resume_batch)
        self.stop_btn.clicked.connect(self.stop_batch)
        self.ocr_selector.currentIndexChanged.connect(self.switch_engine)
        self.save_cfg_btn.clicked.connect(self.save_preprocess_config)
        self.load_cfg_btn.clicked.connect(self.load_preprocess_config)
        self.rotate_preset.currentIndexChanged.connect(self.update_preview)


    # ---------------- CORE LOGIC ----------------
    def switch_engine(self):
        self.engine = {
            "Model - 1": DoctrEngine,
            "Model - 2": EasyOCREngine,
            "Model - 3": PPOCREngine
        }[self.ocr_selector.currentText()]()
        self.output.append("Engine switched")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image")
        if path:
            self.original_image = cv2.imread(path)
            self.update_preview()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_images = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png", ".bmp", ".tiff"))
            ]
            self.output.append(f"Loaded {len(self.folder_images)} images")

    def update_preview(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        if self.enable_pre.isChecked():
            preset = int(self.rotate_preset.currentText().replace("°", ""))
            img = self.engine.preprocess(
                img,
                self.brightness[1].value(),
                self.contrast[1].value() / 100,
                self.gamma[1].value() / 100,
                preset + self.rotate[1].value(),
                self.use_clahe.isChecked()
            )
        self.single_image = img
        self.show_image(img)

    # ---------------- CHAR COUNT ----------------
    def validate_char_count(self, texts):
        val = self.char_count_input.text().strip()
        if not val.isdigit():
            return None

        expected = int(val)

        joined = " ".join(texts)

        tokens = [
            t for t in joined.split()
            if not (len(t) == 1 and t.isalpha())
        ]

        cleaned = re.sub(r'[^A-Za-z0-9]', '', "".join(tokens))

        actual = len(cleaned)
        # ok = expected <= actual
        print("ACTUAL:", actual, "EXPECTED:", expected)
        not_ok = actual<expected

        return not_ok, actual, expected


    def draw_count_status(self, img, not_ok, actual, expected):
        text = f"COUNT Not OK {actual}/{expected}" if not_ok else f"COUNT ok"
        color =  (0, 0, 255) if not_ok else (0, 180, 0)
        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return img

    # ---------------- OCR ----------------
    def run_single(self):
        if self.single_image is None:
            return
        self.output.clear()

        img = self.single_image.copy()
        result = self.engine.run_batch([img])[0]
        raw_text = self.engine.extract_all_text(result)
        regex = self.regex.text().strip()

        matches = []
        if regex:
            matches = self.engine.extract_matches(result, regex)
            img = self.engine.draw_matches(img, result, regex)

        status = self.validate_char_count(raw_text)
        if status:
            img = self.draw_count_status(img, *status)

        self.show_image(img)
        self.output.append("\n".join(matches if regex else raw_text))

    # ---------------- BATCH ----------------
    def start_batch(self):
        if not self.folder_images:
            return
        self.batch_index = 0
        self.batch_json_results = []
        self.batch_running = True
        self.batch_paused = False
        self.run_batch_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.batch_timer.start(250)

    def pause_batch(self):
        self.batch_paused = True
        self.batch_timer.stop()

    def resume_batch(self):
        self.batch_paused = False
        self.batch_timer.start(250)

    def stop_batch(self):
        self.batch_timer.stop()
        self.batch_running = False
        self.run_batch_btn.setEnabled(True)

    def run_batch_step(self):
        if not self.batch_running or self.batch_paused:
            return
        if self.batch_index >= len(self.folder_images):
            self.batch_timer.stop()
            self.batch_running = False
            self.run_batch_btn.setEnabled(True)
            return

        paths = self.folder_images[self.batch_index:self.batch_index + self.BATCH_SIZE]
        images = []

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            if self.enable_pre.isChecked():
                preset = int(self.rotate_preset.currentText().replace("°", ""))
                img = self.engine.preprocess(
                    img,
                    self.brightness[1].value(),
                    self.contrast[1].value() / 100,
                    self.gamma[1].value() / 100,
                    preset + self.rotate[1].value(),
                    self.use_clahe.isChecked()
                )
            images.append(img)

        results = self.engine.run_batch(images)
        for img, res in zip(images, results):
            raw_text = self.engine.extract_all_text(res)
            status = self.validate_char_count(raw_text)
            if status:
                img = self.draw_count_status(img, *status)
            self.show_image(img)

        self.batch_index += self.BATCH_SIZE

    # ---------------- CONFIG ----------------
    def save_preprocess_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "config.json", "JSON (*.json)")
        if not path:
            return
        preset = int(self.rotate_preset.currentText().replace("°", ""))
        config = {
            "ocr_model": self.ocr_selector.currentText(),
            "enable_preprocessing": self.enable_pre.isChecked(),
            "use_clahe": self.use_clahe.isChecked(),
            "brightness": self.brightness[1].value(),
            "contrast": self.contrast[1].value() / 100,
            "gamma": self.gamma[1].value() / 100,
            "fine_rotate": self.rotate[1].value(),
            "rotate_preset": preset
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def load_preprocess_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            cfg = json.load(f)

        idx = self.ocr_selector.findText(cfg.get("ocr_model", "Model - 1"))
        if idx >= 0:
            self.ocr_selector.setCurrentIndex(idx)

        self.enable_pre.setChecked(cfg.get("enable_preprocessing", True))
        self.use_clahe.setChecked(cfg.get("use_clahe", False))
        self.brightness[1].setValue(cfg.get("brightness", 0))
        self.contrast[1].setValue(int(cfg.get("contrast", 1) * 100))
        self.gamma[1].setValue(int(cfg.get("gamma", 1) * 100))
        self.rotate[1].setValue(cfg.get("fine_rotate", 0))
        self.rotate_preset.setCurrentText(f"{cfg.get('rotate_preset',0)}°")
        self.update_preview()

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
