def compute_vertex_jacobians(isle):
    """Area-weighted average of Jacobians per UV vertex."""
    vert_M_sum = {}
    vert_area_sum = {}

    for i, tri in enumerate(isle.tris):
        M = isle.jacobians[i]
        u0, v0 = tri[0]
        u1, v1 = tri[1]
        u2, v2 = tri[2]
        area = abs((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)) * 0.5

        for u, v in tri:
            key = (round(u, 5), round(v, 5))
            if key not in vert_M_sum:
                vert_M_sum[key] = [0.0, 0.0, 0.0, 0.0]
                vert_area_sum[key] = 0.0
            
            vert_M_sum[key][0] += M[0] * area
            vert_M_sum[key][1] += M[1] * area
            vert_M_sum[key][2] += M[2] * area
            vert_M_sum[key][3] += M[3] * area
            vert_area_sum[key] += area
            
    return vert_M_sum, vert_area_sum

from . import stretch_checker
from . import stretch_heatmap


_geo_batch = None
_heatmap_batch = None



def rebuild(props, obj_cache, context):
    """Rebuild the geometry batches from current island data."""
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
            stretch_checker.draw(_geo_batch, opacity, context, use_tint=False)
    elif mode == 'BOTH':
        if _geo_batch:
            stretch_checker.draw(_geo_batch, opacity, context, use_tint=True)


def clear():
    """Release GPU resources. Called on unregister."""
    global _geo_batch, _heatmap_batch
    _geo_batch = None
    _heatmap_batch = None
    stretch_checker.clear()
    stretch_heatmap.clear()
