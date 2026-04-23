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
_geo_batch = None   # position-only TRIS batch (rebuilt on island geometry change)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rebuild(props, obj_cache, context):
    """Rebuild the geometry batch from current island data.

    Called from draw.py whenever island UV geometry has changed (same
    trigger path as padding.rebuild). Zoom does not trigger this rebuild.
    """
    global _geo_batch
    _geo_batch = stretch_checker.build_geometry_batch(obj_cache, props)
    # Phase 4: add heatmap batch here


def draw(props, shader, context):
    """Draw stretch overlay layers.

    'shader' is the outer SMOOTH_COLOR shader (not used by the checker —
    the checker binds its own custom shader). Re-binding the outer shader
    after this call is not needed since stretch is the last draw pass.

    Mode routing:
      BOTH    → checker + heatmap (heatmap stub = no-op until Phase 4)
      CHECKER → checker only
      HEATMAP → heatmap only (no-op until Phase 4)
    """
    mode    = props.stretch_mode
    opacity = props.stretch_opacity

    if mode in ('BOTH', 'CHECKER'):
        # divisions uniform is set fresh each frame from zoom — no rebuild
        stretch_checker.draw(_geo_batch, opacity, context)

    if mode in ('BOTH', 'HEATMAP'):
        pass   # Phase 4


def clear():
    """Release GPU resources. Called on unregister."""
    global _geo_batch
    _geo_batch = None
    stretch_checker.clear()
