# Industrial OCR Evaluation & Extraction Platform

A modular, configurable Industrial OCR application designed for manufacturing, packaging, and quality-inspection environments. The platform enables robust text recognition, custom preprocessing pipelines, and rule-based data extraction using multiple OCR engines—all through a unified interface.

## Overview

Industrial OCR scenarios are challenging due to variable lighting, motion blur, camera noise, diverse fonts/orientations, and strict accuracy requirements for traceability data. This application addresses these challenges by combining:

- Customizable preprocessing pipelines
- Multiple OCR backends (Doctr, EasyOCR, PaddleOCR)
- Regex-based structured data extraction
- Batch and single-image processing
- Config-driven execution for repeatability

## Key Capabilities

1. Multi-Engine OCR Support

	- DocTR — Transformer-based, high accuracy
	- EasyOCR — Lightweight and fast
	- PaddleOCR — Strong multilingual and industrial support

	Each engine can be switched seamlessly without changing application logic.

2. Customizable Preprocessing Pipeline

	- Brightness & contrast adjustment
	- Gamma correction
	- CLAHE (adaptive histogram equalization)
	- Rotation & orientation correction
	- Noise reduction & thresholding

	All preprocessing steps can be saved and loaded via configuration files, enabling repeatable OCR runs across production batches.

3. Regex-Based Intelligent Data Extraction

	- Targeted extraction for Lot/Batch numbers, Manufacturing/Expiry dates, Product codes, Serial numbers, etc.
	- Makes the system suitable for MES/ERP and traceability pipelines.

4. Batch & Single Image Processing

	- Process individual images for tuning and debugging
	- Run folder-level batch OCR for production workloads
	- Pause/resume batch execution
	- Visualize OCR regions and matched patterns

5. Configuration-Driven Workflow

	- Save preprocessing + OCR settings to JSON configuration files
	- Reload configurations to reproduce exact OCR behavior
	- Enables auditable, version-controlled OCR pipelines

## High-Level Architecture

Flow:

Image Input (Camera / Folder) → Preprocessing Pipeline (Configurable) → OCR Engine (Selectable) → Text Detection & Recognition → Regex-Based Data Extraction → Visualization / JSON / CSV Output

## Tech Stack

- Language: Python
- GUI: PyQt
- OCR Engines: DocTR, EasyOCR, PaddleOCR
- Computer Vision: OpenCV
- Acceleration: GPU (CUDA) support
- Data Output: JSON, CSV
- Configuration: JSON-based profiles

## Designed for Industrial Use-Cases

- Manufacturing label inspection
- Packaging verification
- Batch & lot traceability
- Expiry date validation
- OCR model benchmarking
- AI-assisted quality control

## Design Philosophy

- Engine-agnostic — Swap OCR models without rewriting logic
- Config-first — Production reproducibility matters
- Visual debugging — See exactly what the model reads
- Industrial robustness — Built for noisy real-world data

## Future Extensions (Planned)

- PLC / MES integration
- REST API for remote OCR execution
- Automated confidence scoring
- Model ensemble voting
- Edge deployment support

## License

This project is released under an open-source license (add your chosen license here).

## Quick Start

Install dependencies (example):

```bash
pip install PyQt5 opencv-python torch torchvision
pip install python-doctr   # optional: Doctr OCR
pip install easyocr        # optional: EasyOCR
pip install paddlepaddle paddleocr  # optional: PaddleOCR (follow paddle install instructions)
```

Run the application:

```bash
python main.py
```

Use the `OCR Configuration` tab to tune preprocess settings and run single/batch jobs. Use the `OCR Live` tab to load a preprocess JSON and camera config, then start live capture.


