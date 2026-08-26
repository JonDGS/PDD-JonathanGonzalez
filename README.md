# PDD-JonathanGonzalez

Desktop application for detecting, counting, and classifying treetops in aerial images.

The project uses two Ultralytics YOLO models:

1. A detection model locates individual treetops.
2. A classification model assigns a tree type to each detected crop.

The Tkinter interface displays the annotated image, inferred tree count, optional comparison against a YOLO ground-truth label file, and classification totals.

## Project status

This is a research project being restored after a period of inactivity. The current cleanup is intentionally non-destructive: existing models, training runs, plots, and test images remain in place. See [the artifact inventory](docs/artifact-inventory.md) for repository-size details and proposed follow-up.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Tkinter supplied by the operating system's Python installation

Tkinter is part of the Python standard library but is packaged separately by some Linux distributions. It is not the unrelated `tk` package from PyPI.

## Setup

```bash
uv sync
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

Only `.onnx` files appear in the model selectors.

## Tests

```bash
uv run pytest
```

The initial suite covers the pure inference and path-resolution logic extracted from the GUI. Launching and exercising the Tkinter interface requires a graphical environment; model inference itself can also be exercised headlessly in a future integration suite.

## Repository layout

```text
src/tree_top_detector/              Testable application helpers
tests/                              Automated tests
TFG_TreeTopDetector_V2/UI/          Tkinter application and runtime assets
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
- Historical generated outputs and model binaries make the Git repository unusually large.
- The displayed "precision" value is currently the inferred-count/ground-truth-count ratio, not an object-detection precision metric.
