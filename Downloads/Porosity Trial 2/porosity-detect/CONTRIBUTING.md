# Contributing to porosity-detect

Thank you for your interest in contributing to `porosity-detect`! This project aims to advance automated microstructural characterization for aerospace materials using data science and machine learning methods.

## How to Contribute

### Reporting Issues

If you encounter a bug or have a feature request, please open an issue on [GitHub Issues](https://github.com/dinusha-desilva/porosity-detect/issues). When reporting bugs, include:

- A description of the problem
- Steps to reproduce (input image type, parameters used, preset selected)
- Expected vs. actual output
- Python version and OS

### Suggesting Enhancements

We welcome suggestions for new features, especially in these planned areas:

- **Additional material presets** — parameter tuning for new alloy or composite systems
- **Grain size analysis** — automated grain boundary detection and ASTM E112-aligned grain size measurement
- **Fiber volume fraction** — fiber segmentation and Vf quantification for composite cross-sections
- **ML property prediction** — linking microstructural features to mechanical performance

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run the test suite: `pytest tests/`
5. Ensure your code follows the existing style (PEP 8, type hints where practical)
6. Submit a pull request with a clear description of the changes

### Adding Material System Support

If you work with a material system not currently covered by the presets (e.g., ceramic matrix composites, powder metallurgy alloys), contributions of tuned parameter sets are particularly valuable. Include:

- The material system and imaging conditions (magnification, microscope type)
- Parameter values that work well for that system
- Example results on representative images (without proprietary data)

## Development Setup

```bash
git clone https://github.com/dinusha-desilva/porosity-detect.git
cd porosity-detect
pip install -e ".[dev]"
pytest tests/
```

## Code of Conduct

Be respectful and constructive in all interactions. This is a scientific tool — contributions should prioritize accuracy, reproducibility, and clarity.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
