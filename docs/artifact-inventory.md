# Tracked Artifact Inventory

Snapshot taken from the `Development` branch on 2026-08-26 before any artifact cleanup.

No files listed here were removed during the foundation cleanup.

## Current disposition

The 17 selector models formerly under `TFG_TreeTopDetector_V2/UI/modelos/` are preserved byte-for-byte in the [`models-v1.0.0` GitHub Release](https://github.com/JonDGS/PDD-JonathanGonzalez/releases/tag/models-v1.0.0). Their release digests match `models/manifest.toml`, and `scripts/fetch_models.py` restores them to the original dropdown directories. Historical training outputs remain tracked pending a separate preservation decision. Existing Git history has not been rewritten.

## Summary

- Tracked files: **346**
- Tracked file content: **746.7 MiB**
- Files larger than 10 MiB: **26**
- Git LFS configuration: **none**

| Category | Approximate size | Purpose |
| --- | ---: | --- |
| `TFG_TreeTopDetector_V2/runs/` | 532.5 MiB | Detection training runs, checkpoints, ONNX exports, plots, and metrics |
| `TFG_TreeTopDetector_V2/UI/modelos/` | 204.8 MiB | Detection and classification models bundled with the GUI |
| Other artifacts | 7.6 MiB | Validation images, notebooks, and miscellaneous research files |
| `TFG_TreeTopDetector_V2/UI/test/` | 1.4 MiB | Sample images and ground-truth labels used by the GUI |
| Source, configuration, and documentation | 0.4 MiB | Python sources, project metadata, lockfile, and documentation |
| `TFG_TreeTopDetector_V2/UI/saves/` | 0.1 MiB | A committed prediction result |

## Binary formats

| Extension | Files | Approximate size |
| --- | ---: | ---: |
| `.onnx` | 28 | 420.3 MiB |
| `.pt` | 22 | 218.6 MiB |
| `.jpg` | 139 | 90.8 MiB |
| `.png` | 88 | 16.1 MiB |

## Recommended follow-up

Before deleting or rewriting history, classify each artifact into one of these destinations:

1. **Runtime assets:** retain only the selected production detection and classification models needed by the GUI.
2. **Reproducible experiment outputs:** publish with a tagged release, model registry, or external artifact store.
3. **Replaceable generated output:** remove from Git and regenerate from documented commands.
4. **Test fixtures:** keep a deliberately small, documented sample set.

Migrating the existing binary history to Git LFS or removing it with `git filter-repo` would rewrite repository history. That should be a separate, explicitly approved operation after backups and artifact destinations are agreed.
