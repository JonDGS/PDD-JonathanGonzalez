"""Filesystem locations used by the legacy Tkinter application."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """Resolved application paths, independent of the process working directory."""

    ui_directory: Path
    default_image: Path
    detection_models: Path
    classification_models: Path
    original_labels: Path
    runs: Path
    saves: Path

    @classmethod
    def from_ui_directory(cls, ui_directory: Path) -> "AppPaths":
        ui_directory = ui_directory.resolve()
        return cls(
            ui_directory=ui_directory,
            default_image=ui_directory / "miselaneos" / "default_image_bg.png",
            detection_models=ui_directory / "modelos" / "detect",
            classification_models=ui_directory / "modelos" / "classify",
            original_labels=ui_directory / "test" / "labels",
            runs=ui_directory / "runs",
            saves=ui_directory / "saves",
        )
