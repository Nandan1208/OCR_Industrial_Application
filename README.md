Industrial OCR & Barcode Extraction Platform

A modular, configurable industrial-vision application for manufacturing,
packaging, and quality inspection. The platform provides robust OCR and
barcode recognition, customizable preprocessing pipelines, and rule-based
data extraction through a unified GUI.

Overview

Industrial inspection presents challenges such as variable lighting, motion
blur, camera noise, skewed orientations, diverse fonts, and strict accuracy
requirements for traceability and compliance. This application addresses
these issues with configurable preprocessing, multiple OCR/barcode engines,
and rule-based extraction and validation.

Key Features

- Multi-engine OCR: DocTR (transformer-based), EasyOCR (lightweight),
	PaddleOCR (multilingual/industrial). Engines are interchangeable for
	benchmarking and production tuning.
- Industrial barcode recognition: 1D (Code128, Code39, EAN, UPC) and 2D
	(QR, DataMatrix). Supports single-image, batch, and live-camera scanning
	with validation and visual feedback (Match / Mismatch / Not Detected).
- Configurable preprocessing: brightness/contrast, gamma correction, CLAHE,
	rotation/orientation correction, noise reduction, thresholding. Pipelines
	are saved/loaded as JSON for repeatability.
- Regex-based data extraction: configurable rules for lot/batch numbers,
	manufacturing/expiry dates, product codes, and serial numbers. Designed
	to integrate with MES/ERP systems.
- Processing modes: single-image tuning, folder-level batch processing,
	and live camera mode with visualization of OCR regions, extracted text,
	and validation status.
- Configuration-driven workflow for auditable, version-controlled inspection
	pipelines.

High-Level Architecture

Image Input (Camera / Folder)
→ Preprocessing Pipeline (Configurable)
→ OCR Engine / Barcode Engine (Selectable)
→ Text Detection / Barcode Decoding
→ Regex-Based Extraction & Validation
→ Visualization / JSON / CSV Output

Tech Stack

- Language: Python
- GUI: PyQt
- OCR Engines: DocTR, EasyOCR, PaddleOCR (optional)
- Barcode decoding: OpenCV + pyzbar (or equivalent)
- Computer Vision: OpenCV
- Acceleration: optional GPU (CUDA)
- Data output: JSON, CSV

Use Cases

- Manufacturing label inspection
- Packaging verification and anti-mixup checks
- Batch & lot traceability
- Expiry date and serial-number validation
- OCR and barcode engine benchmarking

Design Principles

- Engine-agnostic: swap OCR/barcode engines without changing application logic
- Config-first: save reproducible pipelines for production
- Visual debugging: inspect reads and validation results interactively
- Industrial robustness: tuned for noisy, real-world data

Planned Extensions

- PLC / MES integration
- REST API for remote OCR/barcode execution
- Automated confidence scoring and ensemble voting
- Edge deployment support

License

This project is released under an open-source license. Add your chosen
license in this section.

Quick Start

Install (example):

pip install PyQt5 opencv-python torch torchvision
# Optional OCR engines:
pip install python-doctr    # DocTR (optional)
pip install easyocr         # EasyOCR (optional)
pip install paddlepaddle paddleocr  # PaddleOCR (optional)
pip install pyzbar          # Barcode decoding

Run:

python main.py

Notes

- Use the GUI to load and save JSON configuration profiles for preprocessing
	and extraction rules.
- Use the Barcode Module for single, batch, or live validations.
- For production deployment, create and version your JSON profiles to
	guarantee reproducibility.