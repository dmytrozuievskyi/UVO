import math

COL_BLUE = (0.0, 0.0, 1.0, 1.0)
COL_GRAY = (0.214, 0.214, 0.214, 0.0)
COL_RED  = (1.0, 0.0, 0.0, 1.0)

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


def compute_scale_factors(isle, tex_w, tex_h, target_texel):
    """Returns (scale, scale_u, scale_v). scale is the isotropic factor for BFS."""
    if target_texel > 0:
        scale = target_texel / math.sqrt(tex_w * tex_h)
        scale_u = target_texel / tex_w
        scale_v = target_texel / tex_h
    else:
        scale = math.sqrt(isle.uv_area / isle.surface_area) if isle.surface_area > 0 else 1.0
        aspect = tex_h / tex_w if tex_w > 0 else 1.0
        scale_u = scale * math.sqrt(aspect)
        scale_v = scale / math.sqrt(aspect)
    return scale, scale_u, scale_v


def compute_stretch_metrics(M_avg, scale_u, scale_v):
    """SVD-based area_err and angle_err from a [M00,M01,M10,M11] Jacobian."""
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
    return area_err, angle_err


def lerp_color(c1, c2, t):
    """Linear interpolation of 4-component color tuples."""
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
        c1[3] + (c2[3] - c1[3]) * t
    )


def error_to_color(area_err, angle_err, mode):
    """Compute heat color from stretch errors."""
    sign = 1.0 if area_err >= 0 else -1.0

    if mode == 'checker':
        total_err = area_err + sign * angle_err
        val = max(-1.0, min(1.0, total_err * 0.7))
        mag = abs(val)
        boosted_mag = (mag + math.sqrt(mag)) * 0.5
        val = boosted_mag if val >= 0 else -boosted_mag
    else:
        weight = 0.5
        total_err = sign * (abs(area_err) * (1.0 - weight) + angle_err * weight)
        val = max(-1.0, min(1.0, total_err))

    if val <= 0:
        return lerp_color(COL_GRAY, COL_BLUE, -val)
    else:
        return lerp_color(COL_GRAY, COL_RED, val)


def compute_warped_uvs(isle, vert_M_sum, vert_area_sum, scale):
    """BFS integration + Gauss-Seidel relaxation for checker grid warped UVs."""
    pivot_u = (isle.aabb[0] + isle.aabb[2]) * 0.5
    pivot_v = (isle.aabb[1] + isle.aabb[3]) * 0.5

    adj = {}
    for tri in isle.tris:
        keys = [(round(u, 5), round(v, 5)) for u, v in tri]
        for j in range(3):
            k1, k2 = keys[j], keys[(j+1) % 3]
            if k1 != k2:
                adj.setdefault(k1, set()).add(k2)
                adj.setdefault(k2, set()).add(k1)

    root_key = None
    min_dist = float('inf')
    for k in vert_M_sum:
        dist = (k[0] - pivot_u)**2 + (k[1] - pivot_v)**2
        if dist < min_dist:
            min_dist = dist
            root_key = k

    w_dict = {}
    if root_key:
        w_dict[root_key] = root_key
        queue = [root_key]
        q_idx = 0
        while q_idx < len(queue):
            curr = queue[q_idx]
            q_idx += 1
            w_curr = w_dict[curr]
            area_c = vert_area_sum[curr]
            Mc = [m / area_c for m in vert_M_sum[curr]] if area_c > 1e-8 else [1.0, 0.0, 0.0, 1.0]
            
            for nbr in adj.get(curr, []):
                if nbr not in w_dict:
                    area_n = vert_area_sum[nbr]
                    Mn = [m / area_n for m in vert_M_sum[nbr]] if area_n > 1e-8 else [1.0, 0.0, 0.0, 1.0]
                    M00 = (Mc[0] + Mn[0]) * 0.5 * scale
                    M01 = (Mc[1] + Mn[1]) * 0.5 * scale
                    M10 = (Mc[2] + Mn[2]) * 0.5 * scale
                    M11 = (Mc[3] + Mn[3]) * 0.5 * scale
                    du = nbr[0] - curr[0]
                    dv = nbr[1] - curr[1]
                    w_dict[nbr] = (w_curr[0] + M00*du + M01*dv,
                                   w_curr[1] + M10*du + M11*dv)
                    queue.append(nbr)

    if root_key and len(w_dict) > 1:
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
                    du = curr[0] - nbr[0]
                    dv = curr[1] - nbr[1]
                    edges.append((nbr, M00*du + M01*dv, M10*du + M11*dv))
            if edges:
                adj_targets[curr] = edges

        for _ in range(20):
            for curr, edges in adj_targets.items():
                if curr == root_key:
                    continue
                sum_u = sum_v = 0.0
                for (nbr, tu, tv) in edges:
                    w_nbr = w_dict[nbr]
                    sum_u += w_nbr[0] + tu
                    sum_v += w_nbr[1] + tv
                deg = len(edges)
                w_dict[curr] = (sum_u / deg, sum_v / deg)

    return w_dict


_geo_batch = None
_heatmap_batch = None


def rebuild(props, obj_cache, context):
    """Rebuild the geometry batches from current island data."""
    from . import stretch_checker
    from . import stretch_heatmap
    
    global _geo_batch, _heatmap_batch
    _geo_batch = stretch_checker.build_geometry_batch(obj_cache, props)
    _heatmap_batch = stretch_heatmap.build_geometry_batch(obj_cache, props)


def rebuild_from_worker_data(results):
    """Rebuild geometry batches from pre-computed worker data."""
    from . import stretch_checker
    from . import stretch_heatmap

    global _geo_batch, _heatmap_batch

    all_coords  = []
    all_warped  = []
    all_checker = []
    all_heatmap = []

    for data in results.values():
        all_coords.extend(data['coords'])
        all_warped.extend(data['warped_uvs'])
        all_checker.extend(data['checker_colors'])
        all_heatmap.extend(data['heatmap_colors'])

    _geo_batch     = stretch_checker.build_batch_from_precomputed(all_coords, all_warped, all_checker)
    _heatmap_batch = stretch_heatmap.build_batch_from_precomputed(all_coords, all_heatmap)

def draw(props, shader, context):
    """Draw stretch overlay layers."""
    from . import stretch_checker
    from . import stretch_heatmap

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
    from . import stretch_checker
    from . import stretch_heatmap

    global _geo_batch, _heatmap_batch
    _geo_batch = None
    _heatmap_batch = None
    stretch_checker.clear()
    stretch_heatmap.clear()
