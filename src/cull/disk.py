"""Disk detection and JPEG scanning utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cull.dashboard import DiskDisplayInfo, show_disk_selection

logger = logging.getLogger(__name__)

VOLUMES_ROOT: Path = Path("/Volumes")
SYSTEM_VOLUME_NAMES: frozenset[str] = frozenset(
    {"Macintosh HD", "Preboot", "Recovery", "VM", "Data"}
)
JPEG_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg"})


@dataclass
class DiskInfo:
    """Metadata for a detected external disk."""

    name: str
    mount_point: Path
    size: int
    jpeg_count: int


def _is_system_volume(name: str) -> bool:
    """Return True if name is a known system volume."""
    return name in SYSTEM_VOLUME_NAMES or name.startswith("com.apple.")


def _count_jpegs(mount_point: Path) -> int:
    """Count JPEG files under mount_point."""
    return len(scan_jpegs(mount_point))


def detect_external_disks() -> list[DiskInfo]:
    """Scan /Volumes/ and return non-system disks with JPEG counts."""
    if not VOLUMES_ROOT.exists():
        return []
    disks: list[DiskInfo] = []
    for volume in VOLUMES_ROOT.iterdir():
        if not volume.is_dir() or _is_system_volume(volume.name):
            continue
        jpeg_count = _count_jpegs(volume)
        disks.append(
            DiskInfo(
                name=volume.name,
                mount_point=volume,
                size=0,
                jpeg_count=jpeg_count,
            )
        )
    return disks


def _build_display_info(disk: DiskInfo) -> DiskDisplayInfo:
    """Convert DiskInfo to a dashboard-renderable DiskDisplayInfo."""
    return DiskDisplayInfo(
        name=disk.name,
        mount_point=str(disk.mount_point),
        jpeg_count=disk.jpeg_count,
        size_label=f"{disk.size} B" if disk.size > 0 else "",
    )


def prompt_disk_selection(disks: list[DiskInfo]) -> Path:
    """Show a Rich panel disk menu and return the selected mount_point."""
    if len(disks) == 1:
        logger.info("Auto-selecting only disk: %s", disks[0].name)
        return disks[0].mount_point
    display_items = [_build_display_info(d) for d in disks]
    show_disk_selection(display_items)
    raw = input("  Select disk [1]: ").strip() or "1"
    try:
        choice = int(raw) - 1
        if choice < 0 or choice >= len(disks):
            raise ValueError(f"Selection {choice + 1} out of range [1, {len(disks)}]")
        return disks[choice].mount_point
    except ValueError as e:
        raise ValueError(f"Invalid disk selection: {e}") from e


def scan_jpegs(path: Path) -> list[Path]:
    """Recursively find all JPEG files under path (case-insensitive)."""
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in JPEG_SUFFIXES
    )
