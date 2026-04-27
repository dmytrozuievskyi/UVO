"""stretch_checker.py — Checker grid for the Stretch overlay.

The core problem with a vertex-color approach is that the GPU interpolates
colors across each triangle face, turning every checker cell boundary into
a gradient. The fix is a custom GLSL fragment shader that computes the
checker pattern per pixel — no interpolation, always a hard edge.

Bonus: divisions (zoom level) becomes a uniform updated each frame.
The geometry batch only rebuilds when islands change. No debounce needed.

Layout:
  - _get_shader()              lazy-init the custom GLSL shader (cached)
  - build_geometry_batch()     position-only tri batch from island cache
  - draw(batch, opacity, ctx)  bind shader, set uniforms, draw
  - clear()                    release shader on unregister

Phase 3 upgrade path:
  The vertex shader passes pos.xy as uvCoord to the fragment shader.
  In Phase 3, add a per-triangle Jacobian uniform (or UBO) and transform
  uvCoord by J_inv inside the vertex shader before it reaches the fragment.
"""

import gpu
import math
from gpu_extras.batch import batch_for_shader

# ---------------------------------------------------------------------------
# Zoom levels
# ---------------------------------------------------------------------------
# min(zoom_x, zoom_y) — subdivide only when both axes justify it.

_ZOOM_THRESHOLDS = [2.0, 4.0, 8.0, 16.0]   # boundaries between levels 1–5
_ZOOM_DIVISIONS  = [20, 40, 80, 160, 320]   # grid cells per UV tile per axis


def get_zoom(context):
    space = context.space_data
    zoom  = getattr(space, 'zoom', None)
    if zoom is None:
        return 1.0
    return min(zoom[0], zoom[1])


def get_zoom_level(context):
    z = get_zoom(context)
    for i, threshold in enumerate(_ZOOM_THRESHOLDS):
        if z < threshold:
            return i + 1
    return 5


def get_divisions(zoom_level):
    return _ZOOM_DIVISIONS[max(0, min(zoom_level - 1, 4))]


# ---------------------------------------------------------------------------
# GLSL source
# ---------------------------------------------------------------------------
# The vertex shader passes the UV position through to the fragment shader.
# The UV editor's region matrix already maps UV coordinates to screen space,
# so pos.xy IS the UV coordinate — no separate attribute needed.

_VERT_SRC = """
void main() {
    uvCoord = warpedUV;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

# Per-fragment checker. mod() is used instead of % to avoid negative-modulo
# issues with uvCoord values below zero (islands outside the 0-1 tile).
_FRAG_SRC = """
void main() {
    float d  = float(divisions);
    int iu   = int(mod(floor(uvCoord.x * d), 2.0));
    int iv   = int(mod(floor(uvCoord.y * d), 2.0));
    int cell = (iu + iv) % 2;

    vec3 colDark  = vec3(0.0453, 0.0453, 0.0453);
    vec3 colLight = vec3(0.1008, 0.1008, 0.1008);
    vec3 col = (cell == 1) ? colLight : colDark;
    float alpha = opacity;

    if (both_mode == 1) {
        if (cell == 1) {
            col = vec3(1.0, 1.0, 1.0);
            alpha = opacity * 0.2;
        } else {
            col = vec3(0.0, 0.0, 0.0);
            alpha = opacity * 0.5;
        }
    }

    fragColor = vec4(col, alpha);
}
"""

_shader = None   # cached once per session


def _get_shader():
    global _shader
    if _shader is not None:
        return _shader

    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("stretch_checker_iface")
        vert_out.smooth('VEC2', "uvCoord")

        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewProjectionMatrix")
        info.push_constant('FLOAT', "opacity")
        info.push_constant('INT',   "divisions")
        info.push_constant('INT',   "both_mode")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC2', "warpedUV")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source(_VERT_SRC)
        info.fragment_source(_FRAG_SRC)

        _shader = gpu.shader.create_from_info(info)
        del vert_out, info

    except Exception as e:
        print(f"[UVO] stretch_checker shader compile error: {e}")
        _shader = None

    return _shader


# ---------------------------------------------------------------------------
# Batch — geometry only, rebuilt on island change
# ---------------------------------------------------------------------------

def build_geometry_batch(obj_cache, props):
    """Build a position-only TRIS batch from all island triangles.

    This batch is stable — it only changes when UV geometry changes.
    Zoom level (divisions) is handled as a per-frame uniform, not here.
    """
    if not obj_cache:
        return None

    shader = _get_shader()
    if shader is None:
        return None

    coords = []
    warped_uvs = []

    tex_w = float(props.tex_res_x)
    tex_h = float(props.tex_res_y)
    target_texel = float(props.stretch_target_texel)

    for cache in obj_cache.values():
        islands = cache.get('islands')
        if not islands:
            continue
        for isle in islands:
            if target_texel > 0:
                scale = target_texel / math.sqrt(tex_w * tex_h)
            else:
                scale = math.sqrt(isle.uv_area / isle.surface_area) if isle.surface_area > 0 else 1.0
                
            pivot_u = (isle.aabb[0] + isle.aabb[2]) * 0.5
            pivot_v = (isle.aabb[1] + isle.aabb[3]) * 0.5

            # 1. Area-weighted average of Jacobians per UV vertex
            vert_M_sum = {}
            vert_area_sum = {}

            # We use a 5-decimal rounding to identify shared UV vertices
            for i, tri in enumerate(isle.tris):
                M = isle.jacobians[i]
                u0, v0 = tri[0]
                u1, v1 = tri[1]
                u2, v2 = tri[2]
                
                # 2D area of triangle in UV space
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

            # 2. Build vertex adjacency for BFS integration
            adj = {}
            for tri in isle.tris:
                keys = [(round(u, 5), round(v, 5)) for u, v in tri]
                for j in range(3):
                    k1, k2 = keys[j], keys[(j+1)%3]
                    if k1 != k2:
                        adj.setdefault(k1, set()).add(k2)
                        adj.setdefault(k2, set()).add(k1)

            # 3. Find root vertex closest to center
            root_key = None
            min_dist = float('inf')
            for k in vert_M_sum.keys():
                dist = (k[0] - pivot_u)**2 + (k[1] - pivot_v)**2
                if dist < min_dist:
                    min_dist = dist
                    root_key = k

            # 4. BFS integrate to compute pre-warped coordinates
            w_dict = {}
            if root_key:
                # Start root exactly at its own UV
                w_dict[root_key] = root_key
                queue = [root_key]
                q_idx = 0
                while q_idx < len(queue):
                    curr = queue[q_idx]
                    q_idx += 1
                    w_curr = w_dict[curr]

                    area_c = vert_area_sum[curr]
                    if area_c > 1e-8:
                        Mc = [m / area_c for m in vert_M_sum[curr]]
                    else:
                        Mc = [1.0, 0.0, 0.0, 1.0]

                    for nbr in adj.get(curr, []):
                        if nbr not in w_dict:
                            area_n = vert_area_sum[nbr]
                            if area_n > 1e-8:
                                Mn = [m / area_n for m in vert_M_sum[nbr]]
                            else:
                                Mn = [1.0, 0.0, 0.0, 1.0]

                            # Average Jacobian along the edge, apply target scale
                            M00 = (Mc[0] + Mn[0]) * 0.5 * scale
                            M01 = (Mc[1] + Mn[1]) * 0.5 * scale
                            M10 = (Mc[2] + Mn[2]) * 0.5 * scale
                            M11 = (Mc[3] + Mn[3]) * 0.5 * scale

                            du = nbr[0] - curr[0]
                            dv = nbr[1] - curr[1]

                            # Integrate step
                            w_n_u = w_curr[0] + M00 * du + M01 * dv
                            w_n_v = w_curr[1] + M10 * du + M11 * dv
                            
                            w_dict[nbr] = (w_n_u, w_n_v)
                            queue.append(nbr)

            # 5. Gauss-Seidel Relaxation (Poisson Solver)
            if root_key and len(w_dict) > 1:
                # Precompute target vectors for each edge
                adj_targets = {}
                for curr in w_dict:
                    edges = []
                    area_c = vert_area_sum[curr]
                    Mc = [m / area_c for m in vert_M_sum[curr]] if area_c > 1e-8 else [1.0, 0.0, 0.0, 1.0]

                    for nbr in adj.get(curr, []):
                        if nbr in w_dict:
                            area_n = vert_area_sum[nbr]
                            Mn = [m / area_n for m in vert_M_sum[nbr]] if area_n > 1e-8 else [1.0, 0.0, 0.0, 1.0]

                            M00 = (Mc[0] + Mn[0]) * 0.5 * scale
                            M01 = (Mc[1] + Mn[1]) * 0.5 * scale
                            M10 = (Mc[2] + Mn[2]) * 0.5 * scale
                            M11 = (Mc[3] + Mn[3]) * 0.5 * scale

                            # Target vector is from nbr to curr
                            du = curr[0] - nbr[0]
                            dv = curr[1] - nbr[1]
                            t_u = M00 * du + M01 * dv
                            t_v = M10 * du + M11 * dv
                            
                            edges.append((nbr, t_u, t_v))
                    
                    if edges:
                        adj_targets[curr] = edges

                # Relax for 20 iterations to smooth curl error across all edges
                for _ in range(20):
                    for curr, edges in adj_targets.items():
                        if curr == root_key:
                            continue  # Keep the center pivot pinned to prevent drift
                        
                        sum_u = 0.0
                        sum_v = 0.0
                        for (nbr, tu, tv) in edges:
                            w_nbr = w_dict[nbr]
                            sum_u += w_nbr[0] + tu
                            sum_v += w_nbr[1] + tv
                        
                        deg = len(edges)
                        w_dict[curr] = (sum_u / deg, sum_v / deg)

            # 6. Output batch data
            for i, tri in enumerate(isle.tris):
                for u, v in tri:
                    key = (round(u, 5), round(v, 5))
                    # Fallback to pure affine if vertex is disconnected (should never happen)
                    if key in w_dict:
                        w_u, w_v = w_dict[key]
                    else:
                        w_u, w_v = u, v
                        
                    coords.append((u, v, 0.0))
                    warped_uvs.append((w_u, w_v))

    if not coords:
        return None

    try:
        return batch_for_shader(shader, 'TRIS', {
            "pos": coords,
            "warpedUV": warped_uvs
        })
    except Exception as e:
        import traceback
        print(f"[UVO] stretch_checker batch error: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Draw — called every frame
# ---------------------------------------------------------------------------

_draw_error_printed = False

def draw(batch, opacity, context, both_mode=False):
    """Draw the checker grid."""
    global _draw_error_printed
    if batch is None:
        return

    shader = _get_shader()
    if shader is None:
        return

    zoom_level = get_zoom_level(context)
    divisions  = get_divisions(zoom_level)

    try:
        shader.bind()
        shader.uniform_float("opacity",    opacity)
        shader.uniform_int(  "divisions",  divisions)
        shader.uniform_int(  "both_mode",  1 if both_mode else 0)
        batch.draw(shader)
    except Exception as e:
        if not _draw_error_printed:
            import traceback
            print(f"[UVO] stretch_checker draw error: {e}")
            traceback.print_exc()
            _draw_error_printed = True


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def clear():
    """Release the cached shader on unregister."""
    global _shader
    _shader = None
