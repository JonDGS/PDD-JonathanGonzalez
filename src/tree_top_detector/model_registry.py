"""Model catalog and download support."""

import errno
import hashlib
import os
import re
import secrets
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model declared by the manifest."""

    model_id: str
    task: str
    display_name: str
    asset: str
    target: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelManifest:
    """Versioned collection of downloadable models."""

    version: int
    release_tag: str
    base_url: str
    models: tuple[ModelSpec, ...]


def load_manifest(path: str | Path) -> ModelManifest:
    """Load a model manifest from TOML."""
    manifest_path = Path(path)
    with manifest_path.open("rb") as manifest_file:
        data = tomllib.load(manifest_file)

    models_list = []
    model_ids: set[str] = set()
    assets: set[str] = set()
    targets: set[Path] = set()
    for item in data["models"]:
        if item["id"] in model_ids:
            raise ValueError(f"duplicate model ID: {item['id']}")
        model_ids.add(item["id"])
        target = Path(item["target"])
        if target.is_absolute() or ".." in target.parts:
            raise ValueError(f"target must be a safe relative model path: {target}")
        if target in targets:
            raise ValueError(f"duplicate target: {target}")
        targets.add(target)
        asset = item["asset"]
        if not asset or asset != Path(asset).name or "/" in asset or "\\" in asset:
            raise ValueError(f"asset must be a single filename: {asset}")
        if asset in assets:
            raise ValueError(f"duplicate asset: {asset}")
        assets.add(asset)
        checksum = item["sha256"]
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ValueError(f"sha256 must be a lowercase SHA-256 digest: {checksum}")
        size = item["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"size must be a positive integer: {size}")
        models_list.append(
            ModelSpec(
                model_id=item["id"],
                task=item["task"],
                display_name=item["display_name"],
                asset=asset,
                target=target,
                sha256=checksum,
                size=size,
            )
        )
    models = tuple(models_list)
    return ModelManifest(
        version=data["version"],
        release_tag=data["release_tag"],
        base_url=data["base_url"],
        models=models,
    )


def select_models(
    manifest: ModelManifest,
    *,
    task: str | None = None,
    model_ids: tuple[str, ...] = (),
) -> tuple[ModelSpec, ...]:
    """Select manifest entries by task or explicit model ID."""
    if task is not None:
        return tuple(model for model in manifest.models if model.task == task)
    if model_ids:
        requested = set(model_ids)
        selected = tuple(model for model in manifest.models if model.model_id in requested)
        missing = requested - {model.model_id for model in selected}
        if missing:
            raise ValueError(f"unknown model IDs: {', '.join(sorted(missing))}")
        return selected
    return manifest.models


def selector_options(models: list[str], placeholder: str) -> tuple[str, ...]:
    """Return at least one value for a Tkinter model selector."""
    return tuple(models) if models else (placeholder,)


def _validate_runtime_spec(spec: ModelSpec) -> None:
    target = spec.target
    if target.is_absolute() or ".." in target.parts or target.name in {"", ".", ".."}:
        raise ValueError(f"target must be a safe relative model path: {target}")
    if (
        not spec.asset
        or spec.asset != Path(spec.asset).name
        or "/" in spec.asset
        or "\\" in spec.asset
    ):
        raise ValueError(f"asset must be a single filename: {spec.asset}")
    if re.fullmatch(r"[0-9a-f]{64}", spec.sha256) is None:
        raise ValueError(f"sha256 must be a lowercase SHA-256 digest: {spec.sha256}")
    if not isinstance(spec.size, int) or isinstance(spec.size, bool) or spec.size <= 0:
        raise ValueError(f"size must be a positive integer: {spec.size}")


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure model fetching requires O_NOFOLLOW and O_DIRECTORY")
    required = (os.open, os.mkdir, os.rename, os.unlink)
    if not all(function in os.supports_dir_fd for function in required):
        raise RuntimeError("secure model fetching requires descriptor-relative filesystem APIs")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_destination_parent(destination_root: str | Path, target: Path) -> tuple[int, Path]:
    root = Path(destination_root)
    if root.is_symlink():
        raise ValueError(f"destination root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    flags = _directory_flags()
    try:
        parent_fd = os.open(root, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open destination root: {root}") from error

    try:
        for part in target.parent.parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=parent_fd)
            except FileExistsError:
                pass
            try:
                child_fd = os.open(part, flags, dir_fd=parent_fd)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"destination path contains an unsafe or symlinked component: {part}"
                    ) from error
                raise
            os.close(parent_fd)
            parent_fd = child_fd
        return parent_fd, root.resolve(strict=True) / target
    except BaseException:
        os.close(parent_fd)
        raise


def _matches_spec_at(parent_fd: int, filename: str, spec: ModelSpec) -> bool:
    open_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        file_fd = os.open(filename, open_flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return False
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"destination file must not be a symlink: {filename}") from error
        raise

    with os.fdopen(file_fd, "rb") as model_file:
        file_stat = os.fstat(model_file.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"destination must be a regular file: {filename}")
        if file_stat.st_size != spec.size:
            return False
        digest = hashlib.sha256()
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == spec.sha256


def _create_temp_file_at(parent_fd: int, destination_name: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _ in range(100):
        temporary_name = f".{destination_name}.{secrets.token_hex(12)}.part"
        try:
            return os.open(temporary_name, flags, 0o600, dir_fd=parent_fd), temporary_name
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique temporary model file")


def fetch_model(spec: ModelSpec, base_url: str, destination_root: str | Path) -> Path:
    """Download one model atomically and verify its declared size and checksum."""
    _validate_runtime_spec(spec)
    parent_fd, destination = _open_destination_parent(destination_root, spec.target)
    temporary_name: str | None = None
    try:
        if _matches_spec_at(parent_fd, spec.target.name, spec):
            return destination

        download_url = f"{base_url.rstrip('/')}/{quote(spec.asset, safe='')}"
        digest = hashlib.sha256()
        byte_count = 0
        temporary_fd, temporary_name = _create_temp_file_at(parent_fd, spec.target.name)
        with (
            os.fdopen(temporary_fd, "wb") as output,
            urlopen(download_url, timeout=60) as response,
        ):
            while chunk := response.read(1024 * 1024):
                if byte_count + len(chunk) > spec.size:
                    raise ValueError(
                        f"response exceeds declared size for {spec.model_id}: {spec.size}"
                    )
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)

        if byte_count != spec.size:
            raise ValueError(
                f"size mismatch for {spec.model_id}: expected {spec.size}, got {byte_count}"
            )
        if digest.hexdigest() != spec.sha256:
            raise ValueError(f"checksum mismatch for {spec.model_id}")
        os.rename(
            temporary_name,
            spec.target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        return destination
    finally:
        try:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(parent_fd)
