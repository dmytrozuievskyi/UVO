import gpu
import math
from gpu_extras.batch import batch_for_shader
from . import stretch

_ZOOM_THRESHOLDS = [2.0, 4.0, 8.0, 16.0]   # boundaries between levels 1–5
_ZOOM_DIVISIONS  = [10, 20, 40, 80, 160]    # grid cells per UV tile per axis


def get_zoom(context):
    """Zoom factor relative to 256 px/UV unit, derived from projection matrix."""
    try:
        import gpu
        matrix = gpu.matrix.get_projection_matrix()
        # matrix[0][0] maps UV coordinates to NDC [-1, 1]
        pixels_per_uv = abs(matrix[0][0]) * context.region.width * 0.5
        return max(0.01, pixels_per_uv / 256.0)
    except Exception:
        return 1.0


def get_zoom_level(context):
    z = get_zoom(context)
    for i, threshold in enumerate(_ZOOM_THRESHOLDS):
        if z < threshold:
            return i + 1
    return 5


def get_divisions(zoom_level):
    return _ZOOM_DIVISIONS[max(0, min(zoom_level - 1, 4))]



_VERT_SRC = """
void main() {
    uvCoord = warpedUV;
    heatColor = color;
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

    if (use_tint == 1) {
        float tint = heatColor.a;  // alpha carries how much deviation exists
        // Standard cells are very dark (0.045 / 0.10). 
        // For vibrant colors, target a much brighter color when stretched.
        vec3 targetColor = (cell == 1) ? heatColor.rgb : (heatColor.rgb * 0.5);
        
        // Punch small values by multiplying tint, but hard-cap at 0.75
        float mixFactor = min(tint * 1.5, 0.75);
        col = mix(col, targetColor, mixFactor);
    }
    float alpha = opacity;

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
        vert_out.smooth('VEC4', "heatColor")

        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewProjectionMatrix")
        info.push_constant('FLOAT', "opacity")
        info.push_constant('INT',   "divisions")
        info.push_constant('INT',   "use_tint")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC2', "warpedUV")
        info.vertex_in(2, 'VEC4', "color")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source(_VERT_SRC)
        info.fragment_source(_FRAG_SRC)

        _shader = gpu.shader.create_from_info(info)

    except Exception as e:
        print(f"[UVO] stretch_checker shader compile error: {e}")
        _shader = None

    return _shader


def build_geometry_batch(obj_cache, props):
    """Build a position-only TRIS batch from all island triangles."""
    if not obj_cache:
        return None

    shader = _get_shader()
    if shader is None:
        return None

    coords = []
    warped_uvs = []
    colors = []

    for cache in obj_cache.values():
        islands = cache.get('islands')
        if not islands:
            continue
            
        tex_w = cache.get('tex_w', 1024.0)
        tex_h = cache.get('tex_h', 1024.0)
        target_texel = cache.get('target_texel', 500.0)
        
        for isle in islands:
            if target_texel > 0:
                scale = target_texel / math.sqrt(tex_w * tex_h)
                scale_u = target_texel / tex_w
                scale_v = target_texel / tex_h
            else:
                scale = math.sqrt(isle.uv_area / isle.surface_area) if isle.surface_area > 0 else 1.0
                aspect = tex_h / tex_w if tex_w > 0 else 1.0
                scale_u = scale * math.sqrt(aspect)
                scale_v = scale / math.sqrt(aspect)
                
            pivot_u = (isle.aabb[0] + isle.aabb[2]) * 0.5
            pivot_v = (isle.aabb[1] + isle.aabb[3]) * 0.5

            # 1. Area-weighted average of Jacobians per UV vertex
            vert_M_sum, vert_area_sum = stretch.compute_vertex_jacobians(isle)

            # 1.5. Compute heat colors per vertex
            col_blue = (0.0, 0.0, 1.0, 1.0)
            col_gray = (0.214, 0.214, 0.214, 0.0)
            col_red  = (1.0, 0.0, 0.0, 1.0)
            heat_colors = {}
            for key, area in vert_area_sum.items():
                if area > 1e-8:
                    M_avg = [m / area for m in vert_M_sum[key]]
                else:
                    M_avg = [1.0, 0.0, 0.0, 1.0]

                M00 = M_avg[0] * scale_u
                M01 = M_avg[1] * scale_v
                M10 = M_avg[2] * scale_u
                M11 = M_avg[3] * scale_v

                det_M = M00 * M11 - M01 * M10
                area_stretch = math.sqrt(abs(det_M)) if det_M != 0 else 1.0

                E = (M00 + M11) * 0.5
                F = (M00 - M11) * 0.5
                G = (M10 + M01) * 0.5
                H = (M10 - M01) * 0.5
                Q = math.sqrt(E*E + H*H)
                R = math.sqrt(F*F + G*G)
                s1 = Q + R
                s2 = abs(Q - R)
                
                if abs(s1) < 1e-8 or abs(s2) < 1e-8:
                    angle_stretch = 1.0
                else:
                    angle_stretch = (abs(s1/s2) + abs(s2/s1)) * 0.5

                area_err = math.log2(area_stretch) if area_stretch > 1e-8 else 0.0
                angle_err = math.log2(abs(angle_stretch)) if abs(angle_stretch) > 1e-8 else 0.0

                # Additive area + angle error
                sign = 1.0 if area_err >= 0 else -1.0
                total_err = area_err + sign * angle_err

                val = max(-1.0, min(1.0, total_err * 0.7))
                
                mag = abs(val)
                boosted_mag = (mag + math.sqrt(mag)) * 0.5
                val = boosted_mag if val >= 0 else -boosted_mag

                if val <= 0:
                    t = -val
                    c = (
                        col_gray[0] + (col_blue[0] - col_gray[0]) * t,
                        col_gray[1] + (col_blue[1] - col_gray[1]) * t,
                        col_gray[2] + (col_blue[2] - col_gray[2]) * t,
                        col_gray[3] + (col_blue[3] - col_gray[3]) * t
                    )
                else:
                    t = val
                    c = (
                        col_gray[0] + (col_red[0] - col_gray[0]) * t,
                        col_gray[1] + (col_red[1] - col_gray[1]) * t,
                        col_gray[2] + (col_red[2] - col_gray[2]) * t,
                        col_gray[3] + (col_red[3] - col_gray[3]) * t
                    )
                heat_colors[key] = c

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
                    # Fallback to pure affine if vertex is disconnected
                    if key in w_dict:
                        w_u, w_v = w_dict[key]
                    else:
                        w_u, w_v = u, v
                        
                    coords.append((u, v, 0.0))
                    warped_uvs.append((w_u, w_v))
                    colors.append(heat_colors.get(key, col_gray))

    if not coords:
        return None

    try:
        return batch_for_shader(shader, 'TRIS', {
            "pos": coords,
            "warpedUV": warped_uvs,
            "color": colors
        })
    except Exception as e:
        import traceback
        print(f"[UVO] stretch_checker batch error: {e}")
        traceback.print_exc()
        return None


_draw_error_printed = False

def draw(batch, opacity, context, use_tint=False):
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
        shader.uniform_int(  "use_tint",   1 if use_tint else 0)
        batch.draw(shader)
    except Exception as e:
        if not _draw_error_printed:
            import traceback
            print(f"[UVO] stretch_checker draw error: {e}")
            traceback.print_exc()
            _draw_error_printed = True


def clear():
    """Release the cached shader on unregister."""
    global _shader
    _shader = None
