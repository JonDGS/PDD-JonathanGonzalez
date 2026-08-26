"""Pure inference helpers shared by the GUI and tests."""

from pathlib import Path


def bounding_box_from_yolo(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO box to clamped pixel coordinates."""
    box_width = width * image_width
    box_height = height * image_height
    raw_x1 = int((x_center * image_width) - (box_width / 2))
    raw_y1 = int((y_center * image_height) - (box_height / 2))
    raw_x2 = int(raw_x1 + box_width)
    raw_y2 = int(raw_y1 + box_height)
    return (
        max(0, raw_x1),
        max(0, raw_y1),
        min(image_width, raw_x2),
        min(image_height, raw_y2),
    )


def count_ratio(inferred: int, actual: int | None) -> float | None:
    """Return inferred/actual, or ``None`` when no comparison is possible."""
    if actual is None or actual <= 0:
        return None
    return inferred / actual


def count_label_rows(path: str | Path) -> int | None:
    """Count YOLO annotation rows, returning ``None`` for a missing file."""
    label_path = Path(path)
    if not label_path.is_file():
        return None
    with label_path.open(encoding="utf-8") as labels:
        return sum(1 for _ in labels)
