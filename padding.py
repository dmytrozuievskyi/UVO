import math
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from . import utils

batches = {'ok': None, 'bad': None}
_shader = None


def _get_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    return _shader


_PREC = 5   # decimal places for endpoint key matching

def _rk(p):
    return (round(p[0], _PREC), round(p[1], _PREC))


def _build_contours(segs):
    """Sort unordered (p1, p2) segments into ordered closed loops."""
    if not segs:
        return []

    adj = {}
    for i, (a, b) in enumerate(segs):
        ra, rb = _rk(a), _rk(b)
        adj.setdefault(ra, []).append((rb, a, b, i))
        adj.setdefault(rb, []).append((ra, b, a, i))

    used   = set()
    result = []

    for start_rk in list(adj):
        avail = [e for e in adj[start_rk] if e[3] not in used]
        if not avail:
            continue

        rk_next, p_start, p_next, si = avail[0]
        used.add(si)
        pts    = [p_start, p_next]
        cur_rk = rk_next

        for _ in range(len(segs)):
            if cur_rk == start_rk:
                break
            nxt = [e for e in adj.get(cur_rk, []) if e[3] not in used]
            if not nxt:
                break
            rk2, _, p_n2, si2 = nxt[0]
            used.add(si2)
            pts.append(p_n2)
            cur_rk = rk2

        # Discard open chains — treating them as closed polygons produces
        # incorrect miter offsets at the broken endpoints.
        if cur_rk != start_rk:
            continue

        if len(pts) >= 3:
            if _rk(pts[-1]) == _rk(pts[0]):
                pts.pop()
            if len(pts) >= 3:
                result.append(pts)

    return result


_MITER_CAP = 4.0   # clamps miter scale at sharp concave corners


def _in_tri(px, py, t):
    ax, ay = t[0]; bx, by = t[1]; cx, cy = t[2]
    d1 = (px-bx)*(ay-by) - (ax-bx)*(py-by)
    d2 = (px-cx)*(by-cy) - (bx-cx)*(py-cy)
    d3 = (px-ax)*(cy-ay) - (cx-ax)*(py-ay)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _point_in_island(px, py, tris):
    for t in tris:
        if _in_tri(px, py, t):
            return True
    return False


def _offset_contour(pts, res_x, res_y, pad_px, island_tris):
    """Compute a closed offset contour in UV space.

    Works in pixel space internally for uniform texel offsets regardless of
    aspect ratio. Uses miter joins at each vertex so corners meet cleanly.
    Winding is determined by signed area; a point-in-triangle test on the
    first non-degenerate edge corrects holes regardless of loop winding.
    """
    n = len(pts)

    px = [(p[0] * res_x, p[1] * res_y) for p in pts]

    # Signed area → winding sign for consistent outward normals
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += px[i][0] * px[j][1] - px[j][0] * px[i][1]
    sign = 1.0 if area > 0.0 else -1.0

    # Per-edge outward unit normals in pixel space
    en_x = []
    en_y = []
    for i in range(n):
        j  = (i + 1) % n
        dx = px[j][0] - px[i][0]
        dy = px[j][1] - px[i][1]
        L  = math.sqrt(dx * dx + dy * dy)
        if L < 1e-10:
            en_x.append(0.0)
            en_y.append(0.0)
        else:
            en_x.append( sign * dy / L)
            en_y.append(-sign * dx / L)

    # Verify outward direction: step from first edge midpoint — flip if inside.
    flipped = False
    for i in range(n):
        if abs(en_x[i]) < 1e-10 and abs(en_y[i]) < 1e-10:
            continue
        j    = (i + 1) % n
        mx_u   = ((px[i][0] + px[j][0]) * 0.5) / res_x
        my_v   = ((px[i][1] + px[j][1]) * 0.5) / res_y
        step_u = en_x[i] / res_x * 1e-3
        step_v = en_y[i] / res_y * 1e-3
        if _point_in_island(mx_u + step_u, my_v + step_v, island_tris):
            flipped = True
        break

    if flipped:
        en_x = [-v for v in en_x]
        en_y = [-v for v in en_y]

    # Per-vertex miter offset, capped at _MITER_CAP for sharp concave corners.
    vx = []
    vy = []
    for i in range(n):
        p = (i - 1) % n
        sx = en_x[p] + en_x[i]
        sy = en_y[p] + en_y[i]
        SL = math.sqrt(sx * sx + sy * sy)
        if SL < 1e-10:
            vx.append(en_x[i] * pad_px)
            vy.append(en_y[i] * pad_px)
        else:
            mx_ = sx / SL
            my_ = sy / SL
            d   = en_x[p] * mx_ + en_y[p] * my_
            if abs(d) < 1e-6:
                scale = _MITER_CAP * pad_px
            else:
                scale = min(pad_px / d, _MITER_CAP * pad_px)
            vx.append(mx_ * scale)
            vy.append(my_ * scale)

    result = []
    for i in range(n):
        j  = (i + 1) % n
        p1 = ((px[i][0] + vx[i]) / res_x, (px[i][1] + vy[i]) / res_y)
        p2 = ((px[j][0] + vx[j]) / res_x, (px[j][1] + vy[j]) / res_y)
        result.append((p1, p2))
    return result


def _offset_segs(island, res_x, res_y, pad_px):
    contours = _build_contours(island.boundary_segs)
    result   = []
    for pts in contours:
        if len(pts) >= 3:
            result.extend(_offset_contour(pts, res_x, res_y, pad_px, island.tris))
    return result


_EPS = 1e-9


def _expanded_aabb(island, pad_u, pad_v):
    mn_u, mn_v, mx_u, mx_v = island.aabb
    return (mn_u - pad_u, mn_v - pad_v, mx_u + pad_u, mx_v + pad_v)


def _aabbs_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _segs_cross(a1, a2, b1, b2):
    rx = a2[0]-a1[0]; ry = a2[1]-a1[1]
    sx = b2[0]-b1[0]; sy = b2[1]-b1[1]
    rxs = rx*sy - ry*sx
    if abs(rxs) < _EPS:
        return False
    qpx = b1[0]-a1[0]; qpy = b1[1]-a1[1]
    t = (qpx*sy - qpy*sx) / rxs
    u = (qpx*ry - qpy*rx) / rxs
    return _EPS < t < 1.0-_EPS and _EPS < u < 1.0-_EPS


def _zones_collide(off_a, off_b):
    for a1, a2 in off_a:
        for b1, b2 in off_b:
            if _segs_cross(a1, a2, b1, b2):
                return True
    return False


def _find_bad_islands(islands, res_x, res_y, pad_px, shared_seg_cache=None):
    """Return set of island indices whose padding zones overlap.

    shared_seg_cache: pre-computed {index: segs} reused when display and
    detection resolution match (square textures), avoiding redundant work.
    """
    pad_u = pad_px / res_x
    pad_v = pad_px / res_y
    n     = len(islands)

    # Phase 1: spatial grid cull on expanded AABBs.
    exp_aabbs = [_expanded_aabb(isle, pad_u, pad_v) for isle in islands]

    if n > 1:
        diags = sorted(
            math.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2) for b in exp_aabbs
        )
        cell = max(0.05, min(0.5, diags[len(diags)//2] * 2.0))
    else:
        cell = 0.5

    grid = {}
    for i, ab in enumerate(exp_aabbs):
        cx0 = int(math.floor(ab[0] / cell))
        cy0 = int(math.floor(ab[1] / cell))
        cx1 = int(math.floor(ab[2] / cell))
        cy1 = int(math.floor(ab[3] / cell))
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                grid.setdefault((cx, cy), []).append(i)

    candidates = set()
    for cell_ids in grid.values():
        for a in range(len(cell_ids)):
            for b in range(a + 1, len(cell_ids)):
                ia, ib = cell_ids[a], cell_ids[b]
                pk = (ia, ib) if ia < ib else (ib, ia)
                if pk in candidates:
                    continue
                if _aabbs_overlap(exp_aabbs[ia], exp_aabbs[ib]):
                    candidates.add(pk)

    if not candidates:
        return set()

    # Phase 2: precise offset-segment collision for candidates only.
    off_cache = {}

    def _get(i):
        if shared_seg_cache is not None and i in shared_seg_cache:
            return shared_seg_cache[i]
        if i not in off_cache:
            off_cache[i] = _offset_segs(islands[i], res_x, res_y, pad_px)
        return off_cache[i]

    bad = set()
    for ia, ib in candidates:
        off_a = _get(ia)
        if not off_a:
            continue
        off_b = _get(ib)
        if not off_b:
            continue
        if _zones_collide(off_a, off_b):
            bad.add(ia)
            bad.add(ib)

    return bad


def _get_display_aspect():
    """Return W/H aspect of the current UV editor background image, or 1.0."""
    try:
        space = bpy.context.space_data
        img   = getattr(space, 'image', None) if space else None
        if img and img.size[0] > 0 and img.size[1] > 0:
            return img.size[0] / img.size[1]
    except AttributeError:
        pass
    return 1.0


_PADDING_OPACITY = 0.85


def rebuild(props, obj_cache):
    """Rebuild green/red padding line batches from current island data."""
    global batches

    if not obj_cache:
        batches['ok']  = None
        batches['bad'] = None
        return

    pad_px  = int(props.padding_px)
    res_x   = int(props.tex_res_x)
    res_y   = int(props.tex_res_y)

    # disp_res_y is aspect-adjusted for visual uniformity; detection uses real res_y.
    aspect     = _get_display_aspect()
    disp_res_y = res_y / aspect if aspect > 0 else res_y

    col_ok  = (0.0, 1.0, 0.0, _PADDING_OPACITY)
    col_bad = (1.0, 0.0, 0.0, _PADDING_OPACITY)

    ok_coords,  ok_colors  = [], []
    bad_coords, bad_colors = [], []

    def _add(segs, col, cv, cl):
        for p1, p2 in segs:
            cv += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
            cl += [col, col]

    all_islands = [
        isle
        for cache in obj_cache.values()
        if cache.get('islands')
        for isle in cache['islands']
    ]

    # On square textures disp_res_y == res_y, so display segs double as detection segs.
    disp_segs = {i: _offset_segs(isle, res_x, disp_res_y, pad_px)
                 for i, isle in enumerate(all_islands)}

    square = abs(aspect - 1.0) < 1e-4
    bad_idx = _find_bad_islands(
        all_islands, res_x, res_y, pad_px,
        shared_seg_cache=disp_segs if square else None
    )

    for i, isle in enumerate(all_islands):
        segs = disp_segs[i]
        if i in bad_idx:
            _add(segs, col_bad, bad_coords, bad_colors)
        else:
            _add(segs, col_ok,  ok_coords,  ok_colors)

    utils.log("padding", (
        f"pad={pad_px}px res={res_x}x{res_y} aspect={aspect:.3f} "
        f"ok_segs={len(ok_coords)//2} bad_segs={len(bad_coords)//2}"
    ))

    try:
        shader = _get_shader()

        def _make(coords, colors):
            if not coords:
                return None
            return batch_for_shader(shader, 'LINES', {"pos": coords, "color": colors})

        batches['ok']  = _make(ok_coords,  ok_colors)
        batches['bad'] = _make(bad_coords, bad_colors)

    except Exception as e:
        utils.log("padding", f"batch build error: {e}")
        batches['ok']  = None
        batches['bad'] = None


def clear():
    """Release GPU batches."""
    global _shader
    batches['ok']  = None
    batches['bad'] = None
    _shader        = None