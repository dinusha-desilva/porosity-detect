# porosity-detect

**Hybrid two-pass void/porosity detection for optical micrographs of aerospace materials.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

`porosity-detect` automates the detection, segmentation, and quantification of voids and porosity in cross-sectional optical microscopy images of aerospace composite and metallic materials. It uses a **two-pass detection method with morphological reconstruction** that eliminates shadow artifacts common in single-threshold approaches — a persistent source of inter-operator variability in manual analysis.

This tool supports the goals of the [Materials Genome Initiative (MGI)](https://www.mgi.gov/) by applying data science methods to accelerate aerospace materials characterization.

## Key Features

- **Two-pass detection**: Strict threshold finds void cores, moderate threshold finds candidate regions, morphological reconstruction keeps only candidates connected to cores — automatically rejecting shadow artifacts
- **Material presets**: Tuned parameters for carbon fiber/epoxy composites (high and low magnification), additively manufactured metals, and configurable sensitivity levels
- **ImageJ integration**: Accepts ROI masks exported from ImageJ/FIJI for user-defined analysis regions
- **Quantitative output**: Porosity area fraction, void count, size distribution, shape statistics (circularity, aspect ratio), and spatial metrics
- **Batch processing**: Analyze entire directories of micrographs with summary statistics
- **ML property prediction**: Companion pipeline for predicting mechanical properties from microstructural features

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/porosity-detect.git
cd porosity-detect
pip install numpy scipy scikit-image scikit-learn opencv-python-headless matplotlib Pillow
pip install -e .
```

## Quick Start

### Python API

```python
from porosity_detect.two_pass import TwoPassDetector

detector = TwoPassDetector(preset="composite_high_mag")
result = detector.detect(gray_image)

print(f"Porosity: {result['porosity_pct']:.3f}%")
print(f"Voids: {result['void_count']}")
```

### With ImageJ ROI mask

```bash
python analyze_roi.py micrograph.tif roi_mask.png --preset composite_high_mag
```

### Command line

```bash
porosity-detect demo --output demo_results/
porosity-detect analyze micrograph.tif --output results/
porosity-detect validate --n-images 10
```

## How It Works: Two-Pass Detection

Single-threshold void detection faces an inherent tradeoff: low thresholds miss lighter-edged voids, while high thresholds capture polishing shadows as false positives. The two-pass method solves this:

**Pass 1** (strict threshold): Finds only the darkest pixels — definite void cores. These are seeds.

**Pass 2** (moderate threshold): Finds everything that might be a void, including real void edges and shadow artifacts.

**Morphological Reconstruction**: Labels all candidate regions from Pass 2, then keeps only those physically connected to a Pass 1 seed. Disconnected shadow patches are automatically eliminated.

This encodes the domain knowledge that real voids have dark cores while shadows do not.

## Material Presets

| Preset | Best for |
|--------|----------|
| `composite_high_mag` | High-magnification composite cross-sections with visible fibers |
| `composite_low_mag` | Full-panel cross-sections at lower magnification |
| `am_metal` | Additively manufactured Ti-6Al-4V, IN718 |
| `sensitive` | Maximize detection |
| `conservative` | Minimize false positives |

## Validated Results

Tested on 13 real carbon fiber/epoxy composite optical micrographs:

| Metric | Value |
|--------|-------|
| Images analyzed | 13 |
| Total voids detected | 64 |
| Porosity range | 0.000% - 4.682% |
| Mean porosity | 0.859% +/- 1.503% |

## Package Structure

```
porosity-detect/
├── porosity_detect/          # Core package
│   ├── two_pass.py           # Two-pass detector (primary method)
│   ├── classical.py          # Classical image processing pipeline
│   ├── features.py           # Physics-informed feature extraction
│   ├── ml_model.py           # Random Forest classifier
│   ├── hybrid.py             # Hybrid CV+ML pipeline orchestrator
│   ├── metrics.py            # ASTM-aligned porosity quantification
│   ├── synthetic.py          # Synthetic micrograph generator
│   └── visualization.py      # Publication-quality figures
├── analyze_roi.py            # ImageJ ROI integration
├── annotate.py               # Interactive annotation tool
├── train_real.py             # Train on annotated data
├── impact_ml_pipeline.py     # ML impact prediction pipeline
└── tests/
```

## License

Apache License 2.0

## Citation

```bibtex
@software{porosity_detect,
  title={porosity-detect: Two-Pass Void Detection for Aerospace Material Micrographs},
  author={Dinusha},
  year={2026},
  url={https://github.com/YOUR_USERNAME/porosity-detect}
}
```
