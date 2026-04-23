"""stretch_heatmap.py — Heatmap layer for the Stretch overlay.

Phase 4 stub — returns None until the Jacobian pipeline is in place.

When implemented:
  - Per vertex: collect neighbouring triangles, average area+angle stretch
    weighted by triangle area (from cached Jacobians in stretch.py).
  - Map scalar → Blue (compressed) → Gray (correct) → Red (stretched) ramp.
  - Pass per-vertex RGBA to GPU — interpolation creates smooth gradients.
"""


def build_heatmap_batch(props, obj_cache):
    """Stub. Returns None — heatmap not yet implemented (Phase 4)."""
    return None
