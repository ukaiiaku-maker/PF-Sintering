"""Milestone 12 Section 2: code-path audit demonstrating the new unified
transport mode cannot call the legacy Ostwald spatial removal/addition
kernel during a physical timestep -- a grep-based static check (not just a
runtime assertion) of every module implementing the new physics."""

import ast
from pathlib import Path

import pf_sintering.constrained_eta as constrained_eta
import pf_sintering.surface_transport as surface_transport

_FORBIDDEN_NAMES = {
    "ostwald_substrate",
    "ostwald_substrate_diagnostic",
    "ostwald_removal_only",
    "ostwald_addition_only",
    "ostwald_external_reservoir_step",
    "ostwald_partial_addition_step",
}


def _referenced_names(module) -> set[str]:
    src = Path(module.__file__).read_text()
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_surface_transport_module_never_references_ostwald_kernels():
    names = _referenced_names(surface_transport)
    assert names.isdisjoint(_FORBIDDEN_NAMES)


def test_constrained_eta_module_never_references_ostwald_kernels():
    names = _referenced_names(constrained_eta)
    assert names.isdisjoint(_FORBIDDEN_NAMES)


def test_bc_ops_module_never_references_ostwald_kernels():
    import pf_sintering.bc_ops as bc_ops
    names = _referenced_names(bc_ops)
    assert names.isdisjoint(_FORBIDDEN_NAMES)
