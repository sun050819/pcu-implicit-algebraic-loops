"""Unit tests for aloop core modules."""
import numpy as np
import pytest

from aloop import structures, loopeval
from aloop.solve import pcu
from aloop.data import synth


def test_import_structures():
    """Test that structures module exposes the core data classes."""
    assert hasattr(structures, "Loop")
    assert hasattr(structures, "LoopDB")
    assert hasattr(structures, "BreakpointSet")


def test_import_pcu():
    """Test that PCU solver module imports correctly."""
    assert pcu is not None


def test_import_loopeval():
    """Test that loopeval module imports correctly."""
    assert loopeval is not None


def test_synth_loop_case_structure():
    """Test that synthetic loop-case generation returns a valid (A, bt, params, node_fn) tuple."""
    A, bt, params, node_fn = synth.make_loop_case(6, "single", seed=0)
    assert A.shape == (6, 6)
    assert A.dtype == np.int64
    assert (np.diag(A) == 0).all()          # no self-loops
    assert bt.shape == (6,)
    assert params.shape == (6, 3)
    assert callable(node_fn)


def test_find_loop_nodes_detects_cycle():
    """Test that a generated single-loop case is detected as cyclic (deterministic seed)."""
    A, bt, params, node_fn = synth.make_loop_case(6, "single", seed=0)
    loop_nodes = synth.find_loop_nodes(6, A)
    assert len(loop_nodes) >= 2          # an SCC cycle spans at least two nodes
