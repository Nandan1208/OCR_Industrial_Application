import cv2
import re
import torch
import numpy as np

from doctr.models import ocr_predictor
from doctr.io import DocumentFile

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


# ======================================================
# BASE ENGINE
# ======================================================
class BaseOCREngine:
    def preprocess(self, img, brightness, contrast, gamma, rotate_deg, use_clahe):
        out = img.astype(np.float32)
        out = out * contrast + brightness
        out = np.clip(out, 0, 255)

        gamma = max(gamma, 0.01)
        out = 255 * ((out / 255) ** (1 / gamma))
        out = np.clip(out, 0, 255).astype(np.uint8)

        if use_clahe:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0, (8, 8))
            gray = clahe.apply(gray)
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if rotate_deg % 360 != 0:
            h, w = out.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            out = cv2.warpAffine(out, M, (new_w, new_h))
        return out


# ======================================================
# DOCTR ENGINE
# ======================================================
class DoctrEngine(BaseOCREngine):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Doctr using device: {self.device}")

        self.model = ocr_predictor(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def run_batch(self, images):
        buffers = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", rgb)
            buffers.append(buf.tobytes())

        docs = DocumentFile.from_images(buffers)
        with torch.no_grad():
            result = self.model(docs)

        return result.pages

    def extract_all_text(self, page):
        return [
            word.value
            for block in page.blocks
            for line in block.lines
            for word in line.words
        ]

    def extract_matches(self, page, regex):
        pattern = re.compile(regex, re.IGNORECASE)
        matches = []

        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    clean = word.value.replace(" ", "").replace("-", "")
                    if pattern.fullmatch(clean):
                        matches.append(word.value)

        return matches

    def draw_matches(self, img, page, regex):
        pattern = re.compile(regex, re.IGNORECASE)
        h, w = img.shape[:2]

        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    clean = word.value.replace(" ", "").replace("-", "")
                    if pattern.fullmatch(clean):
                        (x0, y0), (x1, y1) = word.geometry
                        p1 = int(x0 * w), int(y0 * h)
                        p2 = int(x1 * w), int(y1 * h)
                        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
                        cv2.putText(
                            img,
                            word.value,
                            (p1[0], max(p1[1] - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2
                        )
        return img


# ======================================================
# EASY OCR ENGINE
# ======================================================
class EasyOCREngine(BaseOCREngine):
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), quantize=True)

    def run_batch(self, images):
        return [self.reader.readtext(img) for img in images]

    def extract_all_text(self, result):
        return [text for _, text, _ in result]

    def extract_matches(self, result, regex):
        pattern = re.compile(regex, re.IGNORECASE)
        matches = []

        for _, text, _ in result:
            clean = text.replace(" ", "").replace("-", "")
            if pattern.fullmatch(clean):
                matches.append(text)

        return matches

    def draw_matches(self, img, result, regex):
        pattern = re.compile(regex, re.IGNORECASE)
        for box, text, _ in result:
            clean = text.replace(" ", "").replace("-", "")
            if pattern.fullmatch(clean):
                pts = np.array(box).astype(int)
                cv2.polylines(img, [pts], True, (255, 0, 0), 2)
                cv2.putText(
                    img,
                    text,
                    tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
        return img


# ======================================================
# PADDLE OCR ENGINE
# ======================================================
class PPOCREngine(BaseOCREngine):
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en"
            
        )

    def run_batch(self, images):
        return [self.ocr.predict(img) for img in images]

    def extract_all_text(self, result):
        texts = []
        for res in result:
            texts.extend(res.get("rec_texts", []))
        return texts

    def extract_matches(self, result, regex):
        pattern = re.compile(regex, re.IGNORECASE)
        matches = []

        for res in result:
            for text in res.get("rec_texts", []):
                clean = text.replace(" ", "").replace("-", "")
                if pattern.fullmatch(clean):
                    matches.append(text)

        return matches

    def draw_matches(self, img, result, regex):
        

        pattern = re.compile(regex, re.IGNORECASE)

        for res in result:
            texts = res.get("rec_texts", [])

            # ✅ SAFE selection of boxes
            boxes = res.get("rec_boxes", None)
            if boxes is None:
                boxes = res.get("dt_polys", None)

            if boxes is None:
                continue

            for text, box in zip(texts, boxes):
                clean = text.replace(" ", "").replace("-", "")
                if pattern.fullmatch(clean):
                    pts = np.array(box).astype(int)
                    cv2.polylines(img, [pts], True, (255, 0, 0), 2)
                    cv2.putText(
                        img,
                        text,
                        tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

        return img
