# porosity-detect

**Automated void/porosity detection for optical micrographs of aerospace materials.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

`porosity-detect` automates the detection, segmentation, and quantification of voids and porosity in cross-sectional optical micrographs of aerospace composite and metallic materials. It uses a **two-pass detection method with morphological reconstruction** that eliminates shadow artifacts common in single-threshold approaches.

This tool supports the goals of the [Materials Genome Initiative (MGI)](https://www.mgi.gov/) by applying data science methods to accelerate aerospace materials characterization.

## Key Features

- **Desktop GUI** — no coding required, just click buttons
- **Two-pass detection** with morphological reconstruction to eliminate shadow artifacts
- **6 material presets** — tuned for composites, woven fabrics, AM metals
- **Threshold calibration tool** — automatically finds optimal settings for your images
- **ImageJ ROI integration** — or use the entire image without a mask
- **Full-resolution overlay output** — same dimensions as your input image
- **Text report + JSON** — complete results for every analysis

## Installation

```bash
git clone https://github.com/dinusha-desilva/porosity-detect.git
cd porosity-detect
pip install numpy scipy matplotlib Pillow
```

## Quick Start

### GUI (recommended for most users)

```bash
python porosity_gui.py
```

1. Click **Browse Image** — select your micrograph
2. Check **"Use entire image as ROI"** or click **Browse Mask** for an ImageJ ROI mask
3. Select a **preset** from the dropdown
4. Click **ANALYZE**
5. Click **Save Results** to export the overlay, report, and JSON

### Command Line

```bash
# Analyze with ROI mask
python analyze_roi.py image.tif mask.png --preset fabric_cross_section --output results

# Analyze entire image (no mask needed)
python analyze_roi.py image.tif --preset composite_high_mag --output results

# Override thresholds manually
python analyze_roi.py image.tif mask.png --strict 0.22 --moderate 0.35 --min-contrast 0.06 --output results

# With pixel size for physical units
python analyze_roi.py image.tif mask.png --preset fabric_cross_section --pixel-size 5.08 --output results
```

### Python API

```python
from porosity_detect.two_pass import TwoPassDetector

detector = TwoPassDetector(preset="fabric_cross_section")
result = detector.detect(gray_image)

print(f"Porosity: {result['porosity_pct']:.3f}%")
print(f"Voids: {result['void_count']}")
```

## How It Works

Single-threshold void detection faces an inherent tradeoff: low thresholds miss lighter-edged voids, while high thresholds capture polishing shadows as false positives.

**Pass 1** (strict threshold): Finds only the darkest pixels — definite void cores. These are seeds.

**Pass 2** (moderate threshold): Finds everything that might be a void, including shadows.

**Morphological Reconstruction**: Keeps only candidate regions physically connected to a Pass 1 seed. Disconnected shadow patches are automatically eliminated.

## Material Presets

| Preset | Best for |
|--------|----------|
| `composite_high_mag` | High-magnification with visible individual fibers |
| `composite_low_mag` | Lower-magnification panel overview |
| `fabric_cross_section` | Full cross-sections of woven fabric composites |
| `am_metal` | Additively manufactured Ti-6Al-4V, IN718 |
| `sensitive` | Maximum detection |
| `conservative` | Minimum false positives |

## Threshold Calibration

If the presets don't match your ImageJ measurements, use the sweep tool to find optimal thresholds:

```bash
python sweep.py image.tif mask.png --target 3.24
```

This tests thousands of threshold combinations and shows which ones produce porosity closest to your target value. It outputs the exact command to run:

```
BEST MATCH:
  --strict 0.22 --moderate 0.35 --min-area 5 --min-contrast 0.05
  Porosity: 3.241% (target: 3.240%, diff: +0.001%)
```

## Output Files

Each analysis produces:

| File | Description |
|------|-------------|
| `*_void_overlay.png` | Full-resolution image with red voids and green ROI outline |
| `*_report.txt` | Complete text report with all statistics and individual void table |
| `*_results.json` | Machine-readable JSON with all data |
| `*_analysis_plot.png` | Matplotlib summary figure (optional) |

## Supported Image Formats

JPG, JPEG, PNG, TIFF, TIF, BMP — for both the micrograph image and the ROI mask.

## ROI Masks from ImageJ

1. Open your image in ImageJ/FIJI
2. Draw your ROI selection
3. Go to **Edit → Selection → Create Mask**
4. **File → Save As → PNG**

If analyzing the entire image, no mask is needed — check "Use entire image" in the GUI or omit the mask argument on the command line.

## Package Structure

```
porosity-detect/
├── porosity_gui.py           # Desktop GUI (no coding required)
├── analyze_roi.py            # Command-line analysis tool
├── sweep.py                  # Threshold calibration tool
├── porosity_detect/          # Core package
│   ├── two_pass.py           # Two-pass detector with morphological reconstruction
│   ├── classical.py          # Classical image processing pipeline
│   ├── features.py           # Physics-informed feature extraction
│   ├── ml_model.py           # Random Forest classifier
│   ├── hybrid.py             # Hybrid CV+ML pipeline
│   ├── metrics.py            # ASTM-aligned porosity quantification
│   ├── synthetic.py          # Synthetic micrograph generator
│   └── visualization.py      # Publication-quality figures
├── annotate.py               # Interactive void annotation tool
├── train_real.py             # Train classifier on annotated data
└── tests/
```

## License

Apache License 2.0

## Citation

```bibtex
@software{porosity_detect,
  title={porosity-detect: Two-Pass Void Detection for Aerospace Material Micrographs},
  author={Dinusha De Silva},
  year={2026},
  url={https://github.com/dinusha-desilva/porosity-detect}
}
```
