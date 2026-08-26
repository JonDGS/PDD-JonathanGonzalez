from pathlib import Path

from tree_top_detector.paths import AppPaths


def test_app_paths_are_anchored_to_the_ui_directory():
    paths = AppPaths.from_ui_directory(Path("/tmp/tree-ui"))

    assert paths.default_image == Path("/tmp/tree-ui/miselaneos/default_image_bg.png")
    assert paths.detection_models == Path("/tmp/tree-ui/modelos/detect")
    assert paths.classification_models == Path("/tmp/tree-ui/modelos/classify")
    assert paths.original_labels == Path("/tmp/tree-ui/test/labels")
    assert paths.runs == Path("/tmp/tree-ui/runs")
    assert paths.saves == Path("/tmp/tree-ui/saves")
