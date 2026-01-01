import cv2
import json
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit,
    QGroupBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from camera.mv_camera import MVCamera
from ocr_engine import DoctrEngine, EasyOCREngine, PPOCREngine


# ================= THREADS (UNCHANGED) =================
class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        self.log.emit("Camera started")
        while self.running:
            frame = self.camera.capture_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
        self.log.emit("Camera stopped")

    def stop(self):
        self.running = False
        self.wait()


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


# ================= GUI =================
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

        self._apply_styles()
        self._build_ui()
        self._connect_signals()

    # ---------------- STYLES ----------------
    def _apply_styles(self):
        self.setStyleSheet("""
        OCRLiveGui {
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

        QTextEdit {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px;
            color: #111827;
            font-family: Consolas, monospace;
        }
        """)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ===== LEFT PANEL =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        back_btn = QPushButton("Back")
        back_btn.setFixedWidth(120)
        back_btn.clicked.connect(self.back_to_selection.emit)

        self.live_image = QLabel("Camera live feed")
        self.live_image.setAlignment(Qt.AlignCenter)
        self.live_image.setSizePolicy(
            self.live_image.sizePolicy().Expanding,
            self.live_image.sizePolicy().Expanding
        )
        self.live_image.setStyleSheet("""
            background: white;
            border: 2px solid #c7d2fe;
            border-radius: 14px;
            color: #64748b;
            font-size: 14px;
        """)

        left_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        left_layout.addWidget(self.live_image, stretch=1)

        # ===== RIGHT PANEL =====
        right_panel = QWidget()
        right_panel.setFixedWidth(420)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(14)

        # Camera controls
        control_group = QGroupBox("Camera Controls")
        cl = QVBoxLayout(control_group)
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
        log_group = QGroupBox("System Logs")
        ll = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(140)
        ll.addWidget(self.log_console)

        # OCR output
        ocr_group = QGroupBox("OCR Output")
        ol = QVBoxLayout(ocr_group)
        self.ocr_output = QTextEdit()
        self.ocr_output.setReadOnly(True)
        ol.addWidget(self.ocr_output)

        # Assemble right
        scroll_layout.addWidget(control_group)
        scroll_layout.addWidget(log_group)
        scroll_layout.addWidget(ocr_group)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)

        # ===== FINAL =====
        root.addWidget(left_panel, stretch=3)
        root.addWidget(right_panel, stretch=1)

    # ---------------- SIGNALS ----------------
    def _connect_signals(self):
        self.load_preprocess_btn.clicked.connect(self.load_preprocess_json)
        self.load_camera_cfg_btn.clicked.connect(self.load_camera_cfg)
        self.connect_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)

    # ---------------- LOGIC (UNCHANGED) ----------------
    def load_preprocess_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preprocess JSON", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            self.preprocess_cfg = json.load(f)

        model = self.preprocess_cfg.get("ocr_model", "Doctr")
        self.ocr_engine = {
            "Doctr": DoctrEngine,
            "EasyOCR": EasyOCREngine,
            "PaddleOCR": PPOCREngine
        }.get(model, DoctrEngine)()

        self.log_console.append(f"OCR loaded: {model}")

    def load_camera_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Camera Config", "", "Config (*.cfg *.ini *.config)")
        if path:
            self.camera_config = path
            self.log_console.append("Camera config loaded")

    def start_camera(self):
        if not self.camera_config or not self.preprocess_cfg:
            self.log_console.append("Load config first")
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
        self.log_console.append("Stopped")

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
