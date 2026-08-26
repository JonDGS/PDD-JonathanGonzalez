from tree_top_detector.inference import (
    bounding_box_from_yolo,
    count_label_rows,
    count_ratio,
)


def test_bounding_box_from_yolo_converts_and_clamps_coordinates():
    assert bounding_box_from_yolo(
        x_center=0.05,
        y_center=0.05,
        width=0.20,
        height=0.20,
        image_width=100,
        image_height=80,
    ) == (0, 0, 15, 12)


def test_bounding_box_from_yolo_preserves_legacy_fractional_rounding():
    assert bounding_box_from_yolo(
        x_center=0.123,
        y_center=0.123,
        width=0.123,
        height=0.123,
        image_width=640,
        image_height=640,
    ) == (39, 39, 117, 117)


def test_count_ratio_returns_none_without_positive_ground_truth():
    assert count_ratio(inferred=4, actual=0) is None


def test_count_label_rows_preserves_legacy_line_counting(tmp_path):
    labels = tmp_path / "image.txt"
    labels.write_text("0 0.5 0.5 0.1 0.1\n\n0 0.2 0.2 0.1 0.1\n")

    assert count_label_rows(labels) == 3
