# Contributing to AI Checkers

Thank you for your interest in contributing to AI Checkers! This document describes the development setup and contribution process.

## Development Setup

```bash
git clone https://github.com/bonanza127/AIcheckers.git
cd AIcheckers
pip install -r requirements.txt
python app.py
```

## Contribution Process

1. **Check existing issues** - Look at the issue tracker for bugs or features that need work
2. **Open an issue** - If you want to work on something new, open an issue first to discuss
3. **Fork and branch** - Create a feature branch from main (`git checkout -b fix/your-fix`)
4. **Write tests** - All new features should include test cases
5. **Submit a PR** - Reference the issue number in your PR description

## Model Contributions

If you are contributing to the detection model:
- Include accuracy metrics on the test dataset (10,000 images)
- Test against adversarial inputs (JPEG compression, resizing, noise injection)
- Document any changes to feature weights or architecture

## Code Style

- Python: Follow PEP 8
- TypeScript/React: Use ESLint config in the repo
- Commit messages: Use conventional commits (`fix:`, `feat:`, `docs:`, etc.)

## Testing Adversarial Robustness

```bash
# Test JPEG compression robustness
python scripts/test_compression_robustness.py --quality 60,70,80,90

# Test resize robustness
python scripts/test_resize_robustness.py --scales 0.5,0.75,1.0,1.5
```

## Areas Needing Help

- SD3 detection accuracy improvement (see #1)
- Browser extension development (see #3)
- JPEG compression robustness (see #6)
- Internationalization (i18n) support
- Mobile responsive UI improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
