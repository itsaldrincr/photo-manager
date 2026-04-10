"""Source-inspection test: AestheticsPredictorV2Linear uses CLIP vision output."""

import ast
import importlib.util
import pathlib
from typing import Optional


PACKAGE_NAME = "aesthetics_predictor"
V2_MODULE = "v2"
PREDICTOR_CLASS = "AestheticsPredictorV2Linear"
LINEAR_HEAD_ATTR = "layers"


def _locate_v2_source() -> pathlib.Path:
    """Return the filesystem path to aesthetics_predictor/v2.py."""
    try:
        spec = importlib.util.find_spec(f"{PACKAGE_NAME}.{V2_MODULE}")
    except ModuleNotFoundError:
        return pathlib.Path()
    if spec is None or spec.origin is None:
        return pathlib.Path()
    return pathlib.Path(spec.origin)


def _parse_forward_body(source: str) -> Optional[ast.FunctionDef]:
    """Return the AST node for AestheticsPredictorV2Linear.forward, or None."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != PREDICTOR_CLASS:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                return item
    return None


def _has_super_forward_call(func_node: ast.FunctionDef) -> bool:
    """Return True if the forward body calls super().forward(...)."""
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "forward":
            continue
        if isinstance(func.value, ast.Call):
            inner = func.value.func
            if isinstance(inner, ast.Name) and inner.id == "super":
                return True
    return False


def _has_layers_call(func_node: ast.FunctionDef) -> bool:
    """Return True if the forward body calls self.layers(...)."""
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != LINEAR_HEAD_ATTR:
            continue
        if isinstance(func.value, ast.Name) and func.value.id == "self":
            return True
    return False


def test_v2_source_exists() -> None:
    """v2.py is locatable on disk."""
    v2_path = _locate_v2_source()
    if not v2_path.is_file():
        import pytest

        pytest.skip("aesthetics_predictor is not installed in this environment")
    assert v2_path.exists(), f"v2.py not found at {v2_path}"


def test_forward_calls_super_forward() -> None:
    """AestheticsPredictorV2Linear.forward delegates to super().forward(...)."""
    v2_path = _locate_v2_source()
    if not v2_path.is_file():
        import pytest

        pytest.skip("aesthetics_predictor is not installed in this environment")
    source = v2_path.read_text(encoding="utf-8")
    forward_node = _parse_forward_body(source)
    assert forward_node is not None, f"{PREDICTOR_CLASS}.forward not found in AST"
    assert _has_super_forward_call(forward_node), (
        f"{PREDICTOR_CLASS}.forward does not call super().forward(...); "
        "expected CLIPVisionModelWithProjection delegation"
    )


def test_forward_feeds_through_linear_head() -> None:
    """AestheticsPredictorV2Linear.forward passes image_embeds through self.layers."""
    v2_path = _locate_v2_source()
    if not v2_path.is_file():
        import pytest

        pytest.skip("aesthetics_predictor is not installed in this environment")
    source = v2_path.read_text(encoding="utf-8")
    forward_node = _parse_forward_body(source)
    assert forward_node is not None, f"{PREDICTOR_CLASS}.forward not found in AST"
    assert _has_layers_call(forward_node), (
        f"{PREDICTOR_CLASS}.forward does not call self.{LINEAR_HEAD_ATTR}(...); "
        "linear head not consumed in forward pass"
    )
