# PatternMeasure Pro

A computer vision application for measuring paper patterns from photos.

## Features
- Automatic background removal using SegFormer segmentation
- Edge detection and geometric analysis
- User-defined calibration scale
- Automatic edge measurement with annotations
- Area calculation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser at http://localhost:7860

3. Upload a pattern image
4. Set calibration:
   - Enter real-world distance value
   - Select unit (cm, inches, mm)
   - Measure the calibration line in the image (in pixels)
5. Click "Analyze Pattern"

## Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, for faster segmentation)
