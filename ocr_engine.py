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


try:
    from pyzbar.pyzbar import decode as zbar_decode
except ImportError:
    zbar_decode = None
    
import re

def clean_text(text: str) -> str:
    """
    Normalize OCR text for reliable regex matching
    - removes spaces
    - removes symbols
    - keeps only A–Z and 0–9
    """
    return re.sub(r'[^A-Za-z0-9]', '', text)

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
        if not regex:
            return img

        pattern = re.compile(regex, re.IGNORECASE)
        h, w = img.shape[:2]

        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                cleaned = clean_text(line_text)

                if pattern.search(cleaned):
                    for word in line.words:
                        (x0, y0), (x1, y1) = word.geometry
                        p1 = int(x0 * w), int(y0 * h)
                        p2 = int(x1 * w), int(y1 * h)

                        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)

                    cv2.putText(
                        img,
                        line_text,
                        (p1[0], max(p1[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        return img


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

class PPOCREngine(BaseOCREngine):
    def __init__(self):
        self.ocr = PaddleOCR(lang="en", device="gpu", ocr_version="PP-OCRv4")

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
    def normalize_box(self, box):
        box = np.array(box)

        # Case 1: flat array [x1,y1,x2,y2,...]
        if box.ndim == 1 and box.size == 8:
            box = box.reshape(4, 2)

        # Case 2: extra nesting [[[x,y]...]]
        if box.ndim == 3:
            box = box[0]

        # Ensure correct shape
        if box.shape != (4, 2):
            return None

        return box.astype(np.int32)

    def draw_matches(self, img, result, regex):
        pattern = re.compile(regex, re.IGNORECASE)

        for res in result:
            texts = res.get("rec_texts", [])
            boxes = res.get("dt_polys", None)

            if not texts or boxes is None:
                continue

            # Normalize all boxes first
            norm_boxes = []
            for b in boxes:
                pts = self.normalize_box(b)
                if pts is not None:
                    norm_boxes.append(pts)

            # Draw ALL boxes in light color
            for pts in norm_boxes:
                cv2.polylines(img, [pts], True, (180, 180, 180), 1)

            # Highlight matched text
            for text in texts:
                clean = clean_text(text)
                if pattern.search(clean):
                    for pts in norm_boxes:
                        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
                        cv2.putText(
                            img,
                            text,
                            tuple(pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

        return img




    
class BarcodeEngine(BaseOCREngine):
    def __init__(self):
        super().__init__()

    # ---------------- NORMALIZATION ----------------
    def normalize(self, text: str) -> str:
        return (
            text.strip()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

    # ---------------- DECODE ----------------
    def run_batch(self, images):
        outputs = []

        for img in images:
            if img is None:
                outputs.append([])
                continue

            # Alpha-safe
            # if img.ndim == 3 and img.shape[2] == 4:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # # Grayscale
            # if img.ndim == 3:
            #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # else:
            #     gray = img

            # Soft binarization
            # _, gray = cv2.threshold(
            #     gray, 0, 255,
            #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
            # )

            # if np.mean(gray) > 127:
            #     gray = cv2.bitwise_not(gray)

            outputs.append(zbar_decode(img))

        return outputs

    # ---------------- MATCH LOGIC ----------------
    def extract_matches(self, result, expected_input: str):
        matches = []
        values = []

        if not result:
            return matches, values

        expected_clean = (
            self.normalize(expected_input)
            if expected_input else None
        )

        pattern = None
        if expected_clean:
            pattern = re.compile(
                re.escape(expected_clean),
                re.IGNORECASE
            )

        for b in result:
            raw = b.data.decode("utf-8", errors="ignore")
            clean = self.normalize(raw)
            values.append(clean)

            if pattern and pattern.fullmatch(clean):
                matches.append(clean)

        return matches, values

    # ---------------- VISUAL FEEDBACK ----------------
    def draw_matches(self, img, result, expected_input: str):
        matches, values = self.extract_matches(result, expected_input)

        if matches:
            text = f"MATCH : {matches[0]}"
            color = (0, 200, 0)
        else:
            shown = values[0] if values else "NO BARCODE"
            text = f"NOT MATCH : {shown}"
            color = (0, 0, 255)

        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )
        return img