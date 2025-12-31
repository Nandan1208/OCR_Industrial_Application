import cv2
import json
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from camera.mv_camera import MVCamera
from ocr_engine import DoctrEngine, EasyOCREngine, PPOCREngine


# ==================================================
# CAMERA WORKER
# ==================================================
class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        self.log.emit("📷 Camera thread started")
        while self.running:
            frame = self.camera.capture_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
        self.log.emit("⛔ Camera thread stopped")

    def stop(self):
        self.running = False
        self.wait()


# ==================================================
# OCR WORKER
# ==================================================
class OCRWorker(QThread):
    text_ready = pyqtSignal(str)

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

            if texts:
                self.text_ready.emit("\n".join(texts))

    def stop(self):
        self.running = False
        self.wait()


# ==================================================
# OCR LIVE GUI
# ==================================================
class OCRLiveGui(QWidget):
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

    # ==================================================
    def _build_ui(self):
        self.live_image = QLabel("Camera Live Feed")
        self.live_image.setAlignment(Qt.AlignCenter)
        self.live_image.setMinimumSize(800, 600)

        self.load_preprocess_btn = QPushButton("📄 Load Preprocess JSON")
        self.load_camera_cfg_btn = QPushButton("📄 Load Camera Config")
        self.connect_camera_btn = QPushButton("▶ Connect Camera")
        self.stop_camera_btn = QPushButton("⛔ Stop Camera")
        self.stop_camera_btn.setEnabled(False)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)

        self.ocr_output = QTextEdit()
        self.ocr_output.setReadOnly(True)

        controls = QVBoxLayout()
        controls.addWidget(self.load_preprocess_btn)
        controls.addWidget(self.load_camera_cfg_btn)
        controls.addWidget(self.connect_camera_btn)
        controls.addWidget(self.stop_camera_btn)
        controls.addStretch()

        right = QVBoxLayout()
        right.addLayout(controls)
        right.addWidget(QLabel("Logs"))
        right.addWidget(self.log_console)
        right.addWidget(QLabel("OCR Output"))
        right.addWidget(self.ocr_output)

        main = QHBoxLayout(self)
        main.addWidget(self.live_image, 3)
        main.addLayout(right, 1)

    # ==================================================
    def _connect_signals(self):
        self.load_preprocess_btn.clicked.connect(self.load_preprocess_json)
        self.load_camera_cfg_btn.clicked.connect(self.load_camera_cfg)
        self.connect_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)

    # ==================================================
    def load_preprocess_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preprocess JSON", "", "JSON (*.json)"
        )
        if not path:
            return

        with open(path, "r") as f:
            self.preprocess_cfg = json.load(f)

        # ✅ MODEL SELECTION FROM JSON
        model = self.preprocess_cfg.get("ocr_model", "Doctr")
        engine_map = {
            "Doctr": DoctrEngine,
            "EasyOCR": EasyOCREngine,
            "PaddleOCR": PPOCREngine
        }
        self.ocr_engine = engine_map.get(model, DoctrEngine)()

        self.log_console.append(f"🤖 OCR model loaded from JSON: {model}")

    # ==================================================
    def load_camera_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera Config", "", "Config (*.cfg *.ini *.config)"
        )
        if path:
            self.camera_config = path
            self.log_console.append(f"📄 Camera config loaded: {path}")

    # ==================================================
    def start_camera(self):
        if not self.camera_config or not self.preprocess_cfg:
            self.log_console.append("❌ Load camera config and preprocess JSON first")
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
        self.ocr_worker.text_ready.connect(self.ocr_output.append)
        self.ocr_worker.start()

        self.connect_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)

    # ==================================================
    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera.release()
            self.camera_worker = None

        if self.ocr_worker:
            self.ocr_worker.stop()
            self.ocr_worker = None

        self.connect_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.log_console.append("⛔ Camera stopped")

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
