import cv2
import json
import re
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QGroupBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from camera.mv_camera import MVCamera
from ocr_engine import DoctrEngine, EasyOCREngine, PPOCREngine
import os, csv
from datetime import datetime

# ==================================================
# CAMERA THREAD
# ==================================================
class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        self.log.emit("📷 Camera started")
        while self.running:
            frame = self.camera.capture_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
        self.log.emit("⛔ Camera stopped")

    def stop(self):
        self.running = False
        self.wait()


# ==================================================
# OCR THREAD
# ==================================================
class OCRWorker(QThread):
    text_ready = pyqtSignal(list)

    def __init__(self, engine, cfg):
        super().__init__()
        self.engine = engine
        self.cfg = cfg
        self.frame = None
        self.running = True

    def update_frame(self, frame):
        self.frame = frame.copy()

    def run(self):
        while self.running:
            if self.frame is None:
                continue

            img = self.frame
            self.frame = None

            if self.cfg.get("enable_preprocessing", True):
                total_rot = self.cfg.get("rotate_preset", 0) + self.cfg.get("fine_rotate", 0)
                img = self.engine.preprocess(
                    img,
                    self.cfg.get("brightness", 0),
                    self.cfg.get("contrast", 1.0),
                    self.cfg.get("gamma", 1.0),
                    total_rot,
                    self.cfg.get("use_clahe", False)
                )

            result = self.engine.run_batch([img])[0]
            texts = self.engine.extract_all_text(result)
            self.text_ready.emit(texts)

    def stop(self):
        self.running = False
        self.wait()


# ==================================================
# OCR LIVE GUI
# ==================================================
class OCRLiveGui(QWidget):
    back_to_selection = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.camera = None
        self.camera_worker = None
        self.ocr_worker = None

        self.camera_config = None
        self.preprocess_cfg = None
        self.ocr_engine = None

        self.camera_serial = "055060223096"

        self._build_ui()
        self._connect_signals()
        self._init_logger()
        self.log("LIVE OCR started")

        self.live_results = []
        self.frame_counter = 0
        self.live_regex = ""

    # ==================================================
    # UI
    # ==================================================
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        # LEFT
        left = QVBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.live_image = QLabel("Camera Live Feed")
        self.live_image.setAlignment(Qt.AlignCenter)
        self.live_image.setMinimumSize(800, 600)
        self.live_image.setStyleSheet(
            "background:white; border:2px solid #c7d2fe; border-radius:14px;"
        )

        left.addWidget(back_btn, alignment=Qt.AlignLeft)
        left.addWidget(self.live_image)

        # RIGHT
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(420)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)

        # Camera controls
        cam_group = QGroupBox("Camera Controls")
        cl = QVBoxLayout(cam_group)
        self.load_preprocess_btn = QPushButton("Load Preprocess JSON")
        self.load_camera_cfg_btn = QPushButton("Load Camera Config")
        self.connect_camera_btn = QPushButton("Connect Camera")
        self.stop_camera_btn = QPushButton("Stop Camera")
        self.stop_camera_btn.setEnabled(False)

        cl.addWidget(self.load_preprocess_btn)
        cl.addWidget(self.load_camera_cfg_btn)
        cl.addWidget(self.connect_camera_btn)
        cl.addWidget(self.stop_camera_btn)

        # Logs
        log_group = QGroupBox("Logs")
        ll = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(130)
        ll.addWidget(self.log_console)

        # OCR Output
        ocr_group = QGroupBox("OCR Output")
        ol = QVBoxLayout(ocr_group)
        self.ocr_output = QTextEdit()
        self.ocr_output.setReadOnly(True)
        ol.addWidget(self.ocr_output)

        # Count Status
        count_group = QGroupBox("OCR Count Status")
        cll = QVBoxLayout(count_group)
        self.char_count_input = QLineEdit()
        self.char_count_input.setPlaceholderText("Expected character count")
        self.count_status_label = QLabel("Status: —")
        cll.addWidget(self.char_count_input)
        cll.addWidget(self.count_status_label)

        right.addWidget(cam_group)
        right.addWidget(log_group)
        right.addWidget(ocr_group)
        right.addWidget(count_group)
        right.addStretch()

        right_scroll.setWidget(right_widget)

        root.addLayout(left, 3)
        root.addWidget(right_scroll, 1)

    # ==================================================
    # SIGNALS
    # ==================================================
    def _connect_signals(self):
        self.load_preprocess_btn.clicked.connect(self.load_preprocess_json)
        self.load_camera_cfg_btn.clicked.connect(self.load_camera_cfg)
        self.connect_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)

    # ==================================================
    # CONFIG LOAD
    # ==================================================
    def load_preprocess_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocess JSON", "", "JSON (*.json)"
        )
        if not path:
            return

        with open(path, "r") as f:
            self.preprocess_cfg = json.load(f)

        model = self.preprocess_cfg.get("ocr_model", "Model - 1")
        self.ocr_engine = {
            "Model - 1": DoctrEngine,
            "Model - 2": EasyOCREngine,
            "Model - 3": PPOCREngine
        }.get(model, DoctrEngine)()

        # ✅ load regex from JSON (if any)
        self.live_regex = self.preprocess_cfg.get("regex", "").strip()

        self.log_console.append(f"OCR model loaded: {model}")



    def load_camera_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Camera Config", "", "Config (*.cfg *.ini *.config)")
        if path:
            self.camera_config = path
            self.log_console.append("Camera config loaded")

    # ==================================================
    # CAMERA CONTROL
    # ==================================================
    def start_camera(self):
        if not self.camera_config or not self.preprocess_cfg:
            self.log_console.append("Load preprocess JSON & camera config first")
            return

        self.camera = MVCamera(self.camera_serial, self.camera_config)
        ok, msg = self.camera.initialize_camera()
        self.log_console.append(msg)
        if not ok:
            return

        self.camera_worker = CameraWorker(self.camera)
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.log.connect(self.log_console.append)
        self.camera_worker.start()

        self.ocr_worker = OCRWorker(self.ocr_engine, self.preprocess_cfg)
        self.ocr_worker.text_ready.connect(self.handle_ocr_result)
        self.ocr_worker.start()

        self.connect_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)

    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera.release()
            self.camera_worker = None
            self.export_live_csv()

        if self.ocr_worker:
            self.ocr_worker.stop()
            self.ocr_worker = None

        self.connect_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.log_console.append("Camera stopped")

    # ==================================================
    # OCR RESULT HANDLING
    # ==================================================
    def handle_ocr_result(self, texts):
        self.frame_counter += 1

        self.ocr_output.clear()
        self.ocr_output.append("\n".join(texts))

        full_text = " ".join(texts)

        # ---------- REGEX ----------
        regex = self.live_regex
        regex_ok = True
        if regex:
            regex_ok = bool(re.search(regex, full_text))

        # ---------- COUNT ----------
        status = self.validate_char_count(texts)
        if status:
            count_ok, actual, expected = status
        else:
            count_ok = True
            actual = ""
            expected = ""

        final_ok = regex_ok and count_ok

        # ---------- UI ----------
        if final_ok:
            self.count_status_label.setText(
                f"OK | chars={actual}/{expected}" if status else "OK"
            )
            self.count_status_label.setStyleSheet("color: green; font-weight:600;")
        else:
            self.count_status_label.setText(
                f"NOT OK | chars={actual}/{expected}" if status else "NOT OK"
            )
            self.count_status_label.setStyleSheet("color: red; font-weight:600;")

        # ---------- LOG ----------
        self.log(
            f"Frame {self.frame_counter} | "
            f"chars={actual} | "
            f"regex={'OK' if regex_ok else 'FAIL'} | "
            f"result={'OK' if final_ok else 'NOT_OK'}"
        )

        # ---------- CSV (ALL FRAMES) ----------
        self.live_results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame": self.frame_counter,
            "detected_count": actual,
            "expected_count": expected,
            "regex": regex,
            "regex_match": regex_ok,
            "final_result": "OK" if final_ok else "NOT_OK"
        })

    # ==================================================
    # COUNT LOGIC
    # ==================================================
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

        # ✅ SAME RULE AS BATCH / SINGLE
        not_ok = actual < expected

        return (not not_ok), actual, expected



    # ==================================================
    # DISPLAY FRAME
    # ==================================================
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.live_image.width(),
            self.live_image.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.live_image.setPixmap(pix)

        if self.ocr_worker:
            self.ocr_worker.update_frame(frame)
    
    # ==================================================

    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = os.path.join("logs", f"live_ocr_{ts}.txt")

        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(f"LIVE OCR LOG STARTED AT {ts}\n")
            f.write("=" * 60 + "\n")


    def log(self, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"

        self.log_console.append(line)

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    def export_live_csv(self):
        if not self.live_results:
            return

        os.makedirs("exports", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("exports", f"live_ocr_{ts}.csv")

        keys = self.live_results[0].keys()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.live_results)

        self.log(f"📁 Live CSV exported: {path}")


