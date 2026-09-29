"""Non-destructive preflight checks for local dataset preparation commands."""

from pathlib import Path


def validate_dataset_paths(source: Path, output: Path) -> None:
    source = source.resolve()
    destination = output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Source and output paths must not overlap in either direction.")
    if len(destination.parts) <= 2 or (destination / ".git").exists():
        raise ValueError(f"Refusing to replace a root or repository folder: {destination}")
    if output.is_symlink() or getattr(output, "is_junction", lambda: False)():
        raise ValueError("Output folder must not be a symbolic link or junction.")
