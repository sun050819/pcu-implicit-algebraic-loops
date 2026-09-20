# -*- coding: utf-8 -*-
"""Simulink integration: .slx parsing -> block dependency graph -> aloop detection/node selection/solving/loop breaking."""
from .parse_slx import parse_slx, SlxModel, SlxBlock, build_node_fn

__all__ = ["parse_slx", "SlxModel", "SlxBlock", "build_node_fn"]
