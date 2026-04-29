"""stretch.py — Stretch overlay coordinator.

Mirrors the shape of padding.py:
  - Module-level state (geometry batch)
  - rebuild(props, obj_cache, context) — called on island geometry change
  - draw(props, shader, context)       — called every frame from draw_callback
  - clear()                            — releases GPU resources on unregister

Zoom is handled entirely inside stretch_checker.draw() via a per-frame uniform.
No debounce or batch rebuild on zoom needed — the geometry never changes on
zoom, only the 'divisions' shader uniform does.
"""

from . import stretch_checker
from . import stretch_heatmap

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_geo_batch = None       # position-only TRIS batch for checker
_heatmap_batch = None   # color-per-vertex TRIS batch for heatmap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rebuild(props, obj_cache, context):
    """Rebuild the geometry batches from current island data.

    Called from draw.py whenever island UV geometry has changed.
    """
    global _geo_batch, _heatmap_batch
    _geo_batch = stretch_checker.build_geometry_batch(obj_cache, props)
    _heatmap_batch = stretch_heatmap.build_geometry_batch(obj_cache, props)


def draw(props, shader, context):
    """Draw stretch overlay layers.

    Mode routing:
      BOTH    → heatmap drawn solid, checker drawn with transparency on top
      CHECKER → checker only
      HEATMAP → heatmap only
    """
    mode    = props.stretch_mode
    opacity = props.stretch_opacity

    if mode == 'HEATMAP':
        if _heatmap_batch:
            stretch_heatmap.draw(_heatmap_batch, opacity, transparent_gray=False)
    elif mode == 'CHECKER':
        if _geo_batch:
            stretch_checker.draw(_geo_batch, opacity, context)
    elif mode == 'BOTH':
        if _geo_batch:
            # Checker is drawn as a solid base first
            stretch_checker.draw(_geo_batch, opacity, context)
        if _heatmap_batch:
            # Heatmap is drawn on top with transparent neutral areas
            stretch_heatmap.draw(_heatmap_batch, opacity, transparent_gray=True)


def clear():
    """Release GPU resources. Called on unregister."""
    global _geo_batch, _heatmap_batch
    _geo_batch = None
    _heatmap_batch = None
    stretch_checker.clear()
    stretch_heatmap.clear()
