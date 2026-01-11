import os, json, cv2, re
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSlider, QTextEdit,
    QLineEdit, QCheckBox, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QSizePolicy
from ocr_engine import DoctrEngine, EasyOCREngine, PPOCREngine
from datetime import datetime
import csv

class OCRGui(QWidget):
    back_to_selection = pyqtSignal()

    def __init__(self):
        super().__init__()

        # ---------- STATE ----------
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

        self._init_logger()
        self.log("Application started")
        # Set initial window size to fit any screen
        self.showMaximized()  
        self.batch_results = []


    # ================= STYLES =================
    def _apply_styles(self):
        self.setStyleSheet("""
        OCRGui {
            background-color: #f5f6f8;
            font-family: 'Segoe UI';
            font-size: 10pt;
        }
        QPushButton {
            background: #0b2c6b;
            color: white;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: 600;
        }
        QPushButton:hover { background: #093070; }
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
        }
        QLineEdit, QTextEdit, QComboBox {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 6px;
        }
        QTextEdit { font-family: Consolas; }
        QSlider {
            min-height: 22px;
        }
        QSlider::groove:horizontal {
            background: #e5e7eb;
            height: 5px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #0b2c6b;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #093070;
        }
        """)

    # ================= UI =================
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        # ================= TOP =================
        top = QHBoxLayout()
        top.setSpacing(10)

        # ---------- IMAGE PANEL ----------
        image_panel = QVBoxLayout()

        # Top button row
        button_row = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setFixedWidth(90)
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setFixedWidth(90)

        button_row.addWidget(back_btn)
        button_row.addWidget(self.btn_reset)
        button_row.addStretch()

        image_panel.addLayout(button_row)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setStyleSheet(
            "background:white; border:2px dashed #c7d2fe; border-radius:14px;"
        )

        # ---------- UPLOAD CARD ----------
        self.upload_card = QWidget(self.image_label)
        self.upload_card.setMinimumSize(0, 0)
        self.upload_card.setMaximumSize(16777215, 16777215)
        self.upload_card.setStyleSheet("""
            background:white;
            border:2px dashed #c7c7c7;
            border-radius:14px;
        """)

        card_layout = QVBoxLayout(self.upload_card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setSpacing(12)
        card_layout.setContentsMargins(40, 40, 40, 40)

        icon = QLabel("☁")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size:40px; color:#7c8cff;")

        title = QLabel("Select your file")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:14pt; font-weight:600;")

        subtitle = QLabel("png, jpg, pdf supported")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#6b7280;")

        self.btn_upload_img = QPushButton("Browse Image")
        self.btn_upload_folder = QPushButton("Browse Folder")

        for b in (self.btn_upload_img, self.btn_upload_folder):
            b.setFixedSize(180, 40)
            b.setStyleSheet("""
                QPushButton {
                    background:#8da2ff;
                    color:white;
                    border-radius:8px;
                    font-weight:600;
                }
                QPushButton:hover { background:#748bff; }
            """)

        card_layout.addWidget(icon)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(8)
        card_layout.addWidget(self.btn_upload_img)
        card_layout.addWidget(self.btn_upload_folder)

        image_panel.addWidget(self.image_label)

        # ---------- CONTROL PANEL ----------
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        top.addLayout(image_panel, 3)
        top.addLayout(control_panel, 1)

        # Preprocessing
        preprocess_group = QGroupBox("Preprocessing")
        pl = QVBoxLayout(preprocess_group)
        pl.setSpacing(6)
        pl.setContentsMargins(10, 15, 10, 10)

        self.enable_pre = QCheckBox("Enable preprocessing")
        self.enable_pre.setChecked(True)
        self.use_clahe = QCheckBox("Enable CLAHE")

        self.brightness = self.make_slider("Brightness", -100, 100, 0)
        self.contrast = self.make_slider("Contrast", 10, 300, 100)
        self.gamma = self.make_slider("Gamma", 10, 300, 100)
        self.rotate = self.make_slider("Rotate", -90, 90, 0)

        pl.addWidget(self.enable_pre)
        pl.addWidget(self.use_clahe)
        pl.addWidget(self.brightness[0])
        pl.addWidget(self.contrast[0])
        pl.addWidget(self.gamma[0])
        pl.addWidget(self.rotate[0])

        # OCR settings
        ocr_group = QGroupBox("OCR Settings")
        ol = QVBoxLayout(ocr_group)
        self.ocr_selector = QComboBox()
        self.ocr_selector.addItems(["Model - 1", "Model - 2", "Model - 3"])
        self.regex = QLineEdit()
        self.regex.setPlaceholderText("Optional regex")
        self.char_count_input = QLineEdit()
        self.char_count_input.setPlaceholderText("Expected character count")

        ol.addWidget(QLabel("Model"))
        ol.addWidget(self.ocr_selector)
        ol.addWidget(QLabel("Regex"))
        ol.addWidget(self.regex)
        ol.addWidget(QLabel("Expected character count"))
        ol.addWidget(self.char_count_input)

        control_panel.addWidget(preprocess_group)
        control_panel.addWidget(ocr_group)

        

        # ================= BOTTOM =================
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        # OCR Output
        output_group = QGroupBox("OCR Output")
        out_l = QVBoxLayout(output_group)
        self.output = QTextEdit()
        self.output.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred
        )
        self.output.setReadOnly(True)
        # self.output.setMinimumHeight(80)   # Very soft minimum
        # self.output.setMaximumHeight(300)  # Prevent too tall
        out_l.addWidget(self.output)

        # Actions
        action_group = QGroupBox("Actions")
        al = QVBoxLayout(action_group)

        # Buttons
        self.save_cfg_btn = QPushButton("Save config")
        self.load_cfg_btn = QPushButton("Load config")
        self.run_single_btn = QPushButton("Run OCR (Single)")
        self.run_batch_btn = QPushButton("Run Batch")

        for b in (
            self.save_cfg_btn,
            self.load_cfg_btn,
            self.run_single_btn,
            self.run_batch_btn
        ):
            al.addWidget(b)

        # Pause / Resume / Stop row
        ctrl = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")

        for b in (self.pause_btn, self.resume_btn, self.stop_btn):
            b.setEnabled(False)
            ctrl.addWidget(b)

        al.addLayout(ctrl)

# 🔒 HARD lock Actions panel height (AFTER layout is complete)
        action_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        action_group.setMaximumHeight(action_group.sizeHint().height())
        bottom.addWidget(output_group, 3)
        bottom.addWidget(action_group, 1)
        # self.upload_card.show()
        main.addLayout(top, 5)
        main.addLayout(bottom, 2)
        # self.upload_card.hide()



    # ================= HELPERS =================
    def make_slider(self, name, mn, mx, val):
        box = QWidget()
        l = QVBoxLayout(box)
        l.setSpacing(4)
        l.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(f"{name}: {val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)
        slider.setMinimumHeight(22)
        
        slider.valueChanged.connect(
            lambda v: (label.setText(f"{name}: {v}"), self.update_preview())
        )
        
        l.addWidget(label)
        l.addWidget(slider)
        return box, slider

    def _connect_signals(self):
        self.btn_upload_img.clicked.connect(self.load_image)
        self.btn_upload_folder.clicked.connect(self.load_folder)
        self.btn_reset.clicked.connect(self.reset_view)

        self.run_single_btn.clicked.connect(self.run_single)
        self.run_batch_btn.clicked.connect(self.start_batch)

        self.pause_btn.clicked.connect(self.pause_batch)
        self.resume_btn.clicked.connect(self.resume_batch)
        self.stop_btn.clicked.connect(self.stop_batch)

        self.use_clahe.stateChanged.connect(self.update_preview)

        self.save_cfg_btn.clicked.connect(self.save_preprocess_config)
        self.load_cfg_btn.clicked.connect(self.load_preprocess_config)
        self.ocr_selector.currentIndexChanged.connect(self.switch_engine)

    # ================= CORE =================
    def switch_engine(self):
        self.engine = {
            "Model - 1": DoctrEngine,
            "Model - 2": EasyOCREngine,
            "Model - 3": PPOCREngine
        }[self.ocr_selector.currentText()]()
        # self.output.append("Engine switched")
        self.log("Engine switched")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if not path:
            return

        self.original_image = cv2.imread(path)
        if self.original_image is None:
            # self.output.append("❌ Failed to load image")
            self.log("❌ Failed to load image")
            return

        # Reset batch state
        self.folder_images = []
        self.batch_running = False

        # Preview immediately
        self.update_preview()

        # self.output.append(f"🖼 Loaded image: {os.path.basename(path)}")
        self.log(f"🖼 Loaded image: {os.path.basename(path)}")


    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        self.folder_images = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg", ".png", ".bmp", ".tiff"))
        ]

        if not self.folder_images:
            # self.output.append("❌ No valid images found")
            self.log("❌ No valid images found")
            return

        # Clear single-image state
        self.original_image = None
        self.single_image = None

        # Keep upload card visible
        self.image_label.clear()
        self.image_label.setText("Batch loaded\nClick 'Run Batch' to preview images")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.upload_card.show()
        self.upload_card.raise_()

        # self.output.append(f"📂 Loaded {len(self.folder_images)} images")
        self.log(f"📂 Loaded {len(self.folder_images)} images")


    def reset_view(self):
        self.original_image = None
        self.single_image = None
        self.folder_images = []

        self.batch_running = False
        self.batch_paused = False
        self.batch_timer.stop()

        self.image_label.clear()
        self.image_label.setText("")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.upload_card.show()
        self.upload_card.raise_()

        self.output.clear()


    def update_preview(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()

        if self.enable_pre.isChecked():
            img = self.engine.preprocess(
                img,
                self.brightness[1].value(),
                self.contrast[1].value() / 100,
                self.gamma[1].value() / 100,
                self.rotate[1].value(),
                self.use_clahe.isChecked()
            )

        self.single_image = img
        self.show_image(img)


    def run_single(self):
        if self.single_image is None:
            return

        self.output.clear()

        img = self.single_image.copy()

        # OCR
        result = self.engine.run_batch([img])[0]
        raw_text = self.engine.extract_all_text(result)

        regex = self.regex.text().strip()
        matches = []

        # Regex highlighting
        if regex:
            matches = self.engine.extract_matches(result, regex)
            img = self.engine.draw_matches(img, result, regex)

        # Character count validation + overlay
        status = self.validate_char_count(raw_text)
        if status:
            img = self.draw_count_status(img, *status)

        # Show annotated image
        self.show_image(img)

        # Output text
        self.output.append(
            "\n".join(matches if regex else raw_text)
        )



    # ================= BATCH (unchanged logic) =================
    def start_batch(self):
        if not self.folder_images:
            return
        self.batch_results.clear()
        self.upload_card.hide()
        self.batch_index = 0
        self.batch_running = True
        self.batch_paused = False

        self.run_batch_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # self.output.append("▶ Batch started")
        self.log("▶ Batch started")
        self.batch_timer.start(300)


    def pause_batch(self):
        self.batch_paused = True
        self.batch_timer.stop()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        # self.output.append("⏸ Batch paused")
        self.log("⏸ Batch paused")


    def resume_batch(self):
        
        if not self.batch_running:
            return
        self.batch_paused = False
        self.batch_timer.start(300)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        # self.output.append("▶ Batch resumed")
        self.log("▶ Batch resumed")


    def stop_batch(self):
        self.batch_timer.stop()
        self.batch_running = False
        self.batch_paused = False

        self.run_batch_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        # self.output.append("⛔ Batch stopped")
        self.log("⛔ Batch stopped")


    def run_batch_step(self):
        if not self.batch_running or self.batch_paused:
            return

        # ✅ FINAL completion check (ONLY place)
        if self.batch_index >= len(self.folder_images):
            self.batch_timer.stop()
            self.batch_running = False

            self.run_batch_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

            self.log("✅ Batch completed")
            self.export_batch_csv()
            self.batch_results.clear()
            return

        batch_paths = self.folder_images[
            self.batch_index : self.batch_index + self.BATCH_SIZE
        ]

        images = []
        valid_paths = []

        for p in batch_paths:
            img = cv2.imread(p)
            if img is None:
                self.log(f"⚠️ Failed to read image: {os.path.basename(p)}")
                continue

            if self.enable_pre.isChecked():
                img = self.engine.preprocess(
                    img,
                    self.brightness[1].value(),
                    self.contrast[1].value() / 100,
                    self.gamma[1].value() / 100,
                    self.rotate[1].value(),
                    self.use_clahe.isChecked()
                )

            images.append(img)
            valid_paths.append(p)

        if not images:
            self.batch_index += self.BATCH_SIZE
            return

        # Run OCR
        results = self.engine.run_batch(images)

        self.log(
            f"Processing images {self.batch_index + 1} → "
            f"{min(self.batch_index + self.BATCH_SIZE, len(self.folder_images))}"
        )

        for i, res in enumerate(results):
            texts = self.engine.extract_all_text(res)
            file_name = os.path.basename(valid_paths[i])

            eval_data = self.evaluate_result(texts)

            self.batch_results.append({
                "file_name": file_name,
                **eval_data
            })

            self.log(
                f"{file_name} | chars={eval_data['detected_count']} | "
                f"result={eval_data['final_result']}"
            )

        # Preview last image of this batch chunk
        annotated = self.engine.draw_matches(
            images[-1].copy(),
            results[-1],
            self.regex.text().strip()
        )
        self.show_image(annotated)
        self.batch_index += self.BATCH_SIZE

    # ================= CONFIG =================
    def save_preprocess_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "config.json", "JSON (*.json)"
        )
        if not path:
            return

        cfg = {
            "ocr_model": self.ocr_selector.currentText(),
            "enable_preprocessing": self.enable_pre.isChecked(),
            "use_clahe": self.use_clahe.isChecked(),
            "brightness": self.brightness[1].value(),
            "contrast": self.contrast[1].value() / 100,
            "gamma": self.gamma[1].value() / 100,
            "fine_rotate": self.rotate[1].value()
        }
        regex = self.regex.text().strip()
        if regex:
            cfg["regex"] = regex
        char_count = self.char_count_input.text().strip()
        if char_count.isdigit():
            cfg["expected_char_count"] = int(char_count)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        self.log(f"💾 Config saved: {path}")


    def load_preprocess_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            cfg = json.load(f)
        self.enable_pre.setChecked(cfg.get("enable_preprocessing", True))
        self.use_clahe.setChecked(cfg.get("use_clahe", False))
        self.brightness[1].setValue(cfg.get("brightness", 0))
        self.contrast[1].setValue(int(cfg.get("contrast", 1) * 100))
        self.gamma[1].setValue(int(cfg.get("gamma", 1) * 100))
        self.rotate[1].setValue(cfg.get("fine_rotate", 0))
        self.update_preview()

    # ================= DISPLAY =================
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
        self.upload_card.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "upload_card") and self.upload_card.isVisible():
            w = max(0, self.image_label.width() - 40)
            h = max(0, self.image_label.height() - 40)

            self.upload_card.setGeometry(20, 20, w, h)
    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = os.path.join("logs", f"ocr_log_{ts}.txt")

        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(f"OCR LOG STARTED AT {ts}\n")
            f.write("=" * 60 + "\n")
    def log(self, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"

        # UI log
        if hasattr(self, "output"):
            self.output.append(line)
        # self.log.append(line)

        # File log
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def evaluate_result(self, texts):
        full_text = " ".join(texts)
        cleaned = re.sub(r'[^A-Za-z0-9]', '', full_text)
        detected_count = len(cleaned)

        expected_raw = self.char_count_input.text().strip()
        regex = self.regex.text().strip()

        expected = int(expected_raw) if expected_raw.isdigit() else None

        # ✅ SAME RULE AS SINGLE
        not_ok = False
        if expected is not None:
            not_ok = detected_count < expected

        regex_ok = True
        if regex:
            regex_ok = bool(re.search(regex, full_text))

        final_ok = (not not_ok) and regex_ok

        return {
            "detected_count": detected_count,
            "expected_count": expected if expected is not None else "",
            "regex": regex if regex else "",
            "regex_match": regex_ok if regex else "N/A",
            "final_result": "OK" if final_ok else "NOT_OK"
        }

    def export_batch_csv(self):
        if not self.batch_results:
            return

        os.makedirs("exports", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("exports", f"batch_result_{ts}.csv")

        keys = [
                "file_name",
                "expected_count",
                "regex",
                "detected_count",
                "regex_match",
                "final_result"
            ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.batch_results)

        self.log(f"📁 Batch CSV exported: {path}")

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

        not_ok = actual < expected

        return not_ok, actual, expected

    def draw_count_status(self, img, not_ok, actual, expected):
        text = f"COUNT Not OK {actual}/{expected}" if not_ok else f"COUNT OK {actual}/{expected}"
        color = (0, 0, 255) if not_ok else (0, 180, 0)

        cv2.putText(
            img, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3
        )
        return img