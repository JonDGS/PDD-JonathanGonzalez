import hashlib
import signal
from pathlib import Path

import pytest

from tree_top_detector import model_registry
from tree_top_detector.model_registry import (
    ModelManifest,
    ModelSpec,
    fetch_model,
    load_manifest,
    select_models,
    selector_options,
)


def test_load_manifest_parses_model_entries(tmp_path: Path):
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
version = 1
release_tag = "models-v1.0.0"
base_url = "https://example.invalid/models-v1.0.0"

[[models]]
id = "detect-best0"
task = "detection"
display_name = "Detection best0"
asset = "detect-best0.onnx"
target = "models/detect/best0.onnx"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
size = 123
""".strip()
    )

    manifest = load_manifest(manifest_path)

    assert manifest.version == 1
    assert manifest.release_tag == "models-v1.0.0"
    assert manifest.models[0].model_id == "detect-best0"
    assert manifest.models[0].target == Path("models/detect/best0.onnx")


def test_load_manifest_rejects_target_path_traversal(tmp_path: Path):
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
version = 1
release_tag = "models-v1.0.0"
base_url = "https://example.invalid/models-v1.0.0"

[[models]]
id = "detect-best0"
task = "detection"
display_name = "Detection best0"
asset = "detect-best0.onnx"
target = "../../escape.onnx"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
size = 123
""".strip()
    )

    with pytest.raises(ValueError, match="relative model path"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_duplicate_model_ids(tmp_path: Path):
    manifest_path = tmp_path / "manifest.toml"
    entry = """
[[models]]
id = "detect-best0"
task = "detection"
display_name = "Detection best0"
asset = "detect-best0.onnx"
target = "models/detect/best0.onnx"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
size = 123
"""
    manifest_path.write_text(
        """
version = 1
release_tag = "models-v1.0.0"
base_url = "https://example.invalid/models-v1.0.0"
""" + entry + entry.replace("best0.onnx", "best1.onnx")
    )

    with pytest.raises(ValueError, match="duplicate model ID"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_asset_path_traversal(tmp_path: Path):
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
version = 1
release_tag = "models-v1.0.0"
base_url = "https://example.invalid/models-v1.0.0"

[[models]]
id = "detect-best0"
task = "detection"
display_name = "Detection best0"
asset = "../detect-best0.onnx"
target = "models/detect/best0.onnx"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
size = 123
""".strip()
    )

    with pytest.raises(ValueError, match="asset must be a single filename"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    ("sha256", "size", "message"),
    [("not-a-digest", 123, "SHA-256"), ("a" * 64, 0, "positive")],
)
def test_load_manifest_rejects_invalid_integrity_metadata(
    tmp_path: Path, sha256: str, size: int, message: str
):
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        f"""
version = 1
release_tag = "models-v1.0.0"
base_url = "https://example.invalid/models-v1.0.0"

[[models]]
id = "detect-best0"
task = "detection"
display_name = "Detection best0"
asset = "detect-best0.onnx"
target = "models/detect/best0.onnx"
sha256 = "{sha256}"
size = {size}
""".strip()
    )

    with pytest.raises(ValueError, match=message):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    ("duplicate_field", "message"),
    [("asset", "duplicate asset"), ("target", "duplicate target")],
)
def test_load_manifest_rejects_duplicate_asset_or_target(
    tmp_path: Path, duplicate_field: str, message: str
):
    second_asset = "first.onnx" if duplicate_field == "asset" else "second.onnx"
    second_target = (
        "models/detect/first.onnx"
        if duplicate_field == "target"
        else "models/detect/second.onnx"
    )
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        f"""
version = 1
release_tag = "models-v1.0.0"
base_url = "https://example.invalid/models-v1.0.0"

[[models]]
id = "first"
task = "detection"
display_name = "First"
asset = "first.onnx"
target = "models/detect/first.onnx"
sha256 = "{'a' * 64}"
size = 1

[[models]]
id = "second"
task = "detection"
display_name = "Second"
asset = "{second_asset}"
target = "{second_target}"
sha256 = "{'b' * 64}"
size = 1
""".strip()
    )

    with pytest.raises(ValueError, match=message):
        load_manifest(manifest_path)


def test_fetch_model_downloads_and_verifies_asset(tmp_path: Path):
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    payload = b"model-bytes"
    (source_dir / "detect-best0.onnx").write_bytes(payload)
    spec = ModelSpec(
        model_id="detect-best0",
        task="detection",
        display_name="Detection best0",
        asset="detect-best0.onnx",
        target=Path("models/detect/best0.onnx"),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )

    destination = fetch_model(spec, source_dir.as_uri(), tmp_path / "checkout")

    assert destination == tmp_path / "checkout/models/detect/best0.onnx"
    assert destination.read_bytes() == payload


def test_fetch_model_rejects_corrupt_asset_without_leaving_output(tmp_path: Path):
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    payload = b"corrupt-model"
    (source_dir / "detect-best0.onnx").write_bytes(payload)
    spec = ModelSpec(
        model_id="detect-best0",
        task="detection",
        display_name="Detection best0",
        asset="detect-best0.onnx",
        target=Path("models/detect/best0.onnx"),
        sha256="a" * 64,
        size=len(payload),
    )
    destination = tmp_path / "checkout/models/detect/best0.onnx"

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_model(spec, source_dir.as_uri(), tmp_path / "checkout")

    assert not destination.exists()
    assert not destination.with_name("best0.onnx.part").exists()


def test_fetch_model_reuses_an_existing_verified_file(tmp_path: Path):
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    payload = b"model-bytes"
    source = source_dir / "detect-best0.onnx"
    source.write_bytes(payload)
    spec = ModelSpec(
        model_id="detect-best0",
        task="detection",
        display_name="Detection best0",
        asset="detect-best0.onnx",
        target=Path("models/detect/best0.onnx"),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    destination_root = tmp_path / "checkout"
    expected = fetch_model(spec, source_dir.as_uri(), destination_root)
    source.unlink()

    assert fetch_model(spec, source_dir.as_uri(), destination_root) == expected


def test_select_models_filters_by_task():
    detection = ModelSpec(
        "detect-best0",
        "detection",
        "Detection best0",
        "detect-best0.onnx",
        Path("detect/best0.onnx"),
        "a" * 64,
        1,
    )
    classification = ModelSpec(
        "classify-best1",
        "classification",
        "Classification best1",
        "classify-best1.onnx",
        Path("classify/best1.onnx"),
        "b" * 64,
        1,
    )
    manifest = ModelManifest(1, "models-v1.0.0", "https://example.invalid", (detection, classification))

    assert select_models(manifest, task="classification") == (classification,)


def test_selector_options_uses_placeholder_when_no_models_exist():
    assert selector_options([], "No detection models found") == (
        "No detection models found",
    )


def test_fetch_model_rejects_symlinked_destination_component(tmp_path: Path):
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    payload = b"model-bytes"
    (source_dir / "detect-best0.onnx").write_bytes(payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination_root = tmp_path / "checkout"
    destination_root.mkdir()
    (destination_root / "models").symlink_to(outside, target_is_directory=True)
    spec = ModelSpec(
        "detect-best0",
        "detection",
        "Detection best0",
        "detect-best0.onnx",
        Path("models/detect/best0.onnx"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    with pytest.raises(ValueError, match="symlink"):
        fetch_model(spec, source_dir.as_uri(), destination_root)

    assert not (outside / "detect/best0.onnx").exists()


def test_fetch_model_does_not_follow_predictable_partial_symlink(tmp_path: Path):
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    payload = b"model-bytes"
    (source_dir / "detect-best0.onnx").write_bytes(payload)
    destination_root = tmp_path / "checkout"
    destination = destination_root / "models/detect/best0.onnx"
    destination.parent.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"keep-me")
    destination.with_name("best0.onnx.part").symlink_to(victim)
    spec = ModelSpec(
        "detect-best0",
        "detection",
        "Detection best0",
        "detect-best0.onnx",
        Path("models/detect/best0.onnx"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    fetch_model(spec, source_dir.as_uri(), destination_root)

    assert victim.read_bytes() == b"keep-me"
    assert destination.read_bytes() == payload


def test_fetch_model_aborts_when_response_exceeds_declared_size(tmp_path: Path):
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    payload = b"larger-than-declared"
    (source_dir / "detect-best0.onnx").write_bytes(payload)
    spec = ModelSpec(
        "detect-best0",
        "detection",
        "Detection best0",
        "detect-best0.onnx",
        Path("models/detect/best0.onnx"),
        hashlib.sha256(payload[:3]).hexdigest(),
        3,
    )

    with pytest.raises(ValueError, match="exceeds declared size"):
        fetch_model(spec, source_dir.as_uri(), tmp_path / "checkout")


def test_fetch_model_rejects_direct_traversal_before_filesystem_mutation(tmp_path: Path):
    payload = b"model-bytes"
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    (source_dir / "model.onnx").write_bytes(payload)
    destination_root = tmp_path / "checkout"
    outside = tmp_path / "created-outside"
    spec = ModelSpec(
        "direct-spec",
        "detection",
        "Direct spec",
        "model.onnx",
        Path("../created-outside/model.onnx"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    with pytest.raises(ValueError, match="safe relative model path"):
        fetch_model(spec, source_dir.as_uri(), destination_root)

    assert not outside.exists()


def test_fetch_model_does_not_use_path_reopened_temp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = b"model-bytes"
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    (source_dir / "model.onnx").write_bytes(payload)
    destination_root = tmp_path / "checkout"
    detect_dir = destination_root / "models/detect"
    detect_dir.mkdir(parents=True)
    parked_dir = destination_root / "models/detect-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_create_temp_file = model_registry._create_temp_file_at

    def swap_then_open(parent_fd: int, destination_name: str):
        detect_dir.rename(parked_dir)
        detect_dir.symlink_to(outside, target_is_directory=True)
        return original_create_temp_file(parent_fd, destination_name)

    monkeypatch.setattr(model_registry, "_create_temp_file_at", swap_then_open)
    spec = ModelSpec(
        "race-test",
        "detection",
        "Race test",
        "model.onnx",
        Path("models/detect/model.onnx"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    fetch_model(spec, source_dir.as_uri(), destination_root)

    assert not (outside / "model.onnx").exists()
    assert (parked_dir / "model.onnx").read_bytes() == payload


def test_fetch_model_closes_parent_descriptor_when_temp_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = b"model-bytes"
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    (source_dir / "model.onnx").write_bytes(payload)
    opened: dict[str, int] = {}
    original_open_parent = model_registry._open_destination_parent
    original_close = model_registry.os.close
    directory_flags = model_registry._directory_flags()

    def record_open_parent(destination_root, target):
        parent_fd, destination = original_open_parent(destination_root, target)
        opened["parent_fd"] = parent_fd
        return parent_fd, destination

    def fail_unlink(*args, **kwargs):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(model_registry, "_open_destination_parent", record_open_parent)
    monkeypatch.setattr(model_registry, "_directory_flags", lambda: directory_flags)
    monkeypatch.setattr(model_registry.os, "unlink", fail_unlink)
    spec = ModelSpec(
        "cleanup-test",
        "detection",
        "Cleanup test",
        "model.onnx",
        Path("models/detect/model.onnx"),
        "a" * 64,
        len(payload),
    )

    with pytest.raises(PermissionError, match="cleanup denied"):
        fetch_model(spec, source_dir.as_uri(), tmp_path / "checkout")

    parent_fd = opened["parent_fd"]
    try:
        model_registry.os.fstat(parent_fd)
    except OSError as error:
        assert error.errno == model_registry.errno.EBADF
    else:
        original_close(parent_fd)
        pytest.fail("parent descriptor remained open")


def test_fetch_model_rejects_fifo_destination_without_blocking(tmp_path: Path):
    if not hasattr(model_registry.os, "mkfifo"):
        pytest.skip("FIFO test requires POSIX mkfifo")

    payload = b"model-bytes"
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    (source_dir / "model.onnx").write_bytes(payload)
    destination_root = tmp_path / "checkout"
    destination = destination_root / "models/detect/model.onnx"
    destination.parent.mkdir(parents=True)
    model_registry.os.mkfifo(destination)
    spec = ModelSpec(
        "fifo-test",
        "detection",
        "FIFO test",
        "model.onnx",
        Path("models/detect/model.onnx"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )

    def fail_if_blocked(_signum, _frame):
        raise TimeoutError("FIFO open blocked")

    previous_handler = signal.signal(signal.SIGALRM, fail_if_blocked)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with pytest.raises(ValueError, match="regular file"):
            fetch_model(spec, source_dir.as_uri(), destination_root)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
