"""XMP sidecar writer — lxml-based, no ML dependencies."""

from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree
from pydantic import BaseModel

from cull.config import SIDECAR_NAMESPACE_CRS, CullConfig
from cull.models import CropProposal, PhotoDecision

logger = logging.getLogger(__name__)

_NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_NS_X = "adobe:ns:meta/"
_RATING_MAP: dict[str, int] = {
    "keeper": 5,
    "select": 4,
    "uncertain": 3,
    "rejected": 1,
    "duplicate": 1,
}


class _XmpPayload(BaseModel):
    """Internal representation of fields to embed in the XMP file."""

    rotation: float | None
    perspective_vertical: float | None
    crop: CropProposal | None
    rating: int


class SidecarWriteInput(BaseModel):
    """Public input bundle for write_for_decision."""

    decision: PhotoDecision
    config: CullConfig


def write_for_decision(input: SidecarWriteInput) -> Path | None:
    """Write an XMP sidecar next to the source file; return its path or None."""
    if not input.config.is_sidecars:
        return None
    payload = _payload_from_decision(input.decision)
    tree = _build_xmp_tree(payload)
    source = input.decision.photo.path
    out_path = source.with_suffix(".xmp")
    out_path.write_bytes(etree.tostring(tree, xml_declaration=True, encoding="UTF-8", pretty_print=True))
    logger.info("wrote sidecar %s", out_path)
    return out_path


def _payload_from_decision(decision: PhotoDecision) -> _XmpPayload:
    """Extract sidecar-relevant fields from a PhotoDecision."""
    rotation: float | None = None
    perspective_vertical: float | None = None
    crop: CropProposal | None = None
    if decision.stage1 and decision.stage1.geometry:
        rotation = decision.stage1.geometry.tilt_degrees
        perspective_vertical = decision.stage1.geometry.keystone_degrees
    if decision.stage2:
        crop = decision.stage2.crop
    rating = _RATING_MAP.get(decision.decision, 3)
    return _XmpPayload(
        rotation=rotation,
        perspective_vertical=perspective_vertical,
        crop=crop,
        rating=rating,
    )


def _build_xmp_tree(payload: _XmpPayload) -> etree._Element:
    """Build the lxml element tree for the XMP document."""
    xmpmeta = etree.Element(
        f"{{{_NS_X}}}xmpmeta",
        nsmap={"x": _NS_X, "rdf": _NS_RDF, "crs": SIDECAR_NAMESPACE_CRS},
    )
    rdf = etree.SubElement(xmpmeta, f"{{{_NS_RDF}}}RDF")
    desc = etree.SubElement(rdf, f"{{{_NS_RDF}}}Description")
    desc.set(f"{{{_NS_RDF}}}about", "")
    _append_crs_fields(desc, payload)
    return xmpmeta


def _append_crs_fields(desc: etree._Element, payload: _XmpPayload) -> None:
    """Append crs: namespace fields onto the RDF Description element."""
    crs = SIDECAR_NAMESPACE_CRS
    desc.set(f"{{{crs}}}Rating", str(payload.rating))
    if payload.rotation is not None:
        desc.set(f"{{{crs}}}StraightenAngle", str(payload.rotation))
    if payload.perspective_vertical is not None:
        desc.set(f"{{{crs}}}PerspectiveVertical", str(payload.perspective_vertical))
    if payload.crop is not None:
        _append_crop_fields(desc, payload.crop)


def _append_crop_fields(desc: etree._Element, crop: CropProposal) -> None:
    """Append crs: crop fields onto the RDF Description element."""
    crs = SIDECAR_NAMESPACE_CRS
    desc.set(f"{{{crs}}}CropTop", str(crop.top))
    desc.set(f"{{{crs}}}CropLeft", str(crop.left))
    desc.set(f"{{{crs}}}CropBottom", str(crop.bottom))
    desc.set(f"{{{crs}}}CropRight", str(crop.right))
