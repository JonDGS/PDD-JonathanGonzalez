# PDD-JonathanGonzalez

Desktop application for detecting, counting, and classifying treetops in aerial images.

The project uses two Ultralytics YOLO models:

1. A detection model locates individual treetops.
2. A classification model assigns a tree type to each detected crop.

The Tkinter interface displays the annotated image, inferred tree count, optional comparison against a YOLO ground-truth label file, and classification totals.

## Project status

This is a research project being restored after a period of inactivity. Selectable ONNX models are preserved as versioned [GitHub Release assets](https://github.com/JonDGS/PDD-JonathanGonzalez/releases/tag/models-v1.0.0), while historical training runs, plots, and test images remain available for later review. See [the artifact inventory](docs/artifact-inventory.md) for the original repository-size snapshot and follow-up notes.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Tkinter supplied by the operating system's Python installation

Tkinter is part of the Python standard library but is packaged separately by some Linux distributions. It is not the unrelated `tk` package from PyPI.

## Setup

```bash
uv sync
uv run python scripts/fetch_models.py --all
```

The fetch command restores all 17 checksum-verified selector models from the `models-v1.0.0` GitHub Release. Existing files with the expected size and SHA-256 digest are reused. The hardened downloader currently requires POSIX descriptor-relative filesystem APIs (`dir_fd`, `O_NOFOLLOW`, and `O_DIRECTORY`), available on Linux and macOS. On Windows, download the release assets manually into the model directories until an equivalent reparse-point-safe implementation is added.

To inspect or download a smaller subset:

```bash
# Show model IDs, tasks, sizes, and display names
uv run python scripts/fetch_models.py --list

# Download only one task
uv run python scripts/fetch_models.py --task detection
uv run python scripts/fetch_models.py --task classification

# Download one or more named models
uv run python scripts/fetch_models.py --model detection-best3
```

The default environment contains the desktop application and test dependencies. Optional toolsets can be installed when needed:

```bash
# Model benchmarking and plots
uv sync --extra analysis

# cx_Freeze packaging support
uv sync --extra build
```

## Run the application

From the repository root:

```bash
uv run python TFG_TreeTopDetector_V2/UI/UI.py
```

Application resources are resolved relative to `TFG_TreeTopDetector_V2/UI/`, so the command no longer depends on launching Python from that directory.

Detection models are loaded from:

```text
TFG_TreeTopDetector_V2/UI/modelos/detect/
```

Classification models are loaded from:

```text
TFG_TreeTopDetector_V2/UI/modelos/classify/
```

Only `.onnx` files appear in the model selectors. Their filenames and local directories are unchanged after fetching, so the dropdown workflow remains the same.

## Tests

```bash
uv run pytest
```

The initial suite covers the pure inference and path-resolution logic extracted from the GUI. Launching and exercising the Tkinter interface requires a graphical environment; model inference itself can also be exercised headlessly in a future integration suite.

## Repository layout

```text
src/tree_top_detector/              Testable application and model-fetch helpers
tests/                              Automated tests
scripts/fetch_models.py             Checksum-verified model downloader
models/manifest.toml                Versioned selector-model catalog
TFG_TreeTopDetector_V2/UI/          Tkinter application and downloaded runtime assets
TFG_TreeTopDetector_V2/utils/       Dataset and model-evaluation utilities
TFG_TreeTopDetector_V2/runs/        Historical training outputs
docs/artifact-inventory.md          Binary and generated-artifact inventory
pyproject.toml                      Dependencies and tool configuration
uv.lock                             Reproducible dependency lockfile
```

## Training and analysis

The repository contains scripts for detection training, classification training, validation, crop extraction, and ONNX model benchmarking. Several still reference historical local dataset paths. Treat them as research scripts until those datasets and commands are documented and parameterized.

The model benchmark utility additionally requires the `analysis` extra:

```bash
uv run --extra analysis python TFG_TreeTopDetector_V2/utils/testModels.py --help
```

## Known limitations

- The GUI remains a single large Tkinter class and should be separated incrementally.
- Training scripts contain environment-specific dataset paths.
- Model inference is synchronous and can block the GUI while processing.
- Historical generated training outputs still make the Git repository unusually large; the selector models have moved to Release assets, but existing Git history has not been rewritten.
- The displayed "precision" value is currently the inferred-count/ground-truth-count ratio, not an object-detection precision metric.
