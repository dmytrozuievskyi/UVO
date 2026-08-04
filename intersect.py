import time
import math
from collections import Counter as _Counter

try:
    from . import utils
except ImportError:
    import utils

EPSILON    = 1e-6
UV_EPS     = 1e-4
UV_DECIMAL = 3


class Island:
    __slots__ = ('tris', 'aabb', 'uv_key', 'local_key', 'ref_a', 'ref_b', 'color', 'object_name',
                 'boundary_segs', 'tri_centers', 'jacobians', 'uv_area', 'surface_area',
                 'face_normals', 'face_areas', 'grad_u', 'grad_v')

    def __init__(self, tris, color, object_name=''):
        self.tris          = tris
        self.color         = color
        self.object_name   = object_name
        self.uv_key        = None
        self.local_key     = None
        self.ref_a         = (0.0, 0.0)
        self.ref_b         = (0.0, 0.0)
        self.boundary_segs = []
        self.jacobians     = []
        self.uv_area       = 0.0
        self.surface_area  = 0.0
        self.face_normals  = []
        self.face_areas    = []
        self.grad_u        = []
        self.grad_v        = []

        if tris:
            all_u = [v[0] for t in tris for v in t]
            all_v = [v[1] for t in tris for v in t]
            self.aabb = (min(all_u), min(all_v), max(all_u), max(all_v))
            self.tri_centers = [
                ((t[0][0]+t[1][0]+t[2][0]) / 3.0,
                 (t[0][1]+t[1][1]+t[2][1]) / 3.0)
                for t in tris
            ]
        else:
            self.aabb        = (0.0, 0.0, 0.0, 0.0)
            self.tri_centers = []


def _uvclose(a, b):
    return abs(a.x - b.x) < UV_EPS and abs(a.y - b.y) < UV_EPS


def _build_uv_adjacency(bm, uv_layer):
    """Build face adjacency by UV connectivity. O(total_loops) pass."""
    fe_to_loop = {}
    for face in bm.faces:
        for loop in face.loops:
            fe_to_loop[(face.index, loop.edge.index)] = loop

    adj = {f.index: [] for f in bm.faces}
    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue
        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        eidx   = edge.index
        l1 = fe_to_loop.get((f1.index, eidx))
        l2 = fe_to_loop.get((f2.index, eidx))
        if l1 is None or l2 is None:
            continue
        uv1a, uv1b = l1[uv_layer].uv, l1.link_loop_next[uv_layer].uv
        uv2a, uv2b = l2[uv_layer].uv, l2.link_loop_next[uv_layer].uv
        if ((_uvclose(uv1a, uv2a) and _uvclose(uv1b, uv2b)) or
                (_uvclose(uv1a, uv2b) and _uvclose(uv1b, uv2a))):
            adj[f1.index].append(f2.index)
            adj[f2.index].append(f1.index)
    return adj


def _extract_boundary_segs(island_faces, face_index_set, uv_layer, uv_adj):
    """Collect UV edges forming the outer contour of the island."""
    segs = []
    for face in island_faces:
        uv_nbrs = uv_adj.get(face.index, [])
        for loop in face.loops:
            other_in_island = [
                lf.index for lf in loop.edge.link_faces
                if lf.index != face.index and lf.index in face_index_set
            ]
            if not other_in_island:
                is_boundary = True
            else:
                is_boundary = not any(fi in uv_nbrs for fi in other_in_island)

            if is_boundary:
                p1 = (loop[uv_layer].uv.x, loop[uv_layer].uv.y)
                p2 = (loop.link_loop_next[uv_layer].uv.x,
                      loop.link_loop_next[uv_layer].uv.y)
                segs.append((p1, p2))
    return segs


def extract_islands(bm_copy, uv_layer, alpha_val, obj_seed, utils_mod,
                    object_name='', matrix_world=None):
    t0 = time.perf_counter()
    bm_copy.faces.ensure_lookup_table()
    uv_adj = _build_uv_adjacency(bm_copy, uv_layer)
    t1 = time.perf_counter()
    utils_mod.log("timing_extract", f"build_uv_adj: {(t1-t0)*1000:.1f}ms")

    # Flood-fill to get total island count before colouring.
    face_groups = []
    visited     = set()

    for seed_face in bm_copy.faces:
        if seed_face.index in visited:
            continue

        face_index_set = set()
        stack = [seed_face]
        visited.add(seed_face.index)
        face_index_set.add(seed_face.index)

        while stack:
            curr = stack.pop()
            for nb_idx in uv_adj.get(curr.index, []):
                if nb_idx not in visited:
                    visited.add(nb_idx)
                    face_index_set.add(nb_idx)
                    stack.append(bm_copy.faces[nb_idx])

        face_groups.append(face_index_set)

    total = len(face_groups)
    t2 = time.perf_counter()
    utils_mod.log("timing_extract", f"flood_fill: {(t2-t1)*1000:.1f}ms")

    islands = []
    t_fan = 0.0
    t_key = 0.0
    t_bound = 0.0

    for idx, face_index_set in enumerate(face_groups):
        col = utils_mod.get_distinct_color(
            idx, total, seed_offset=obj_seed, alpha=alpha_val
        )

        island_faces = [bm_copy.faces[i] for i in face_index_set]
        
        ta = time.perf_counter()
        f_tris, f_jacs, uv_area, surf_area, f_norms, f_areas, g_u, g_v = _fan_tris_and_data(island_faces, uv_layer, matrix_world)
        if not f_tris:
            continue
            
        isle = Island(f_tris, col, object_name)
        isle.jacobians = f_jacs
        isle.uv_area = uv_area
        isle.surface_area = surf_area
        isle.face_normals = f_norms
        isle.face_areas = f_areas
        isle.grad_u = g_u
        isle.grad_v = g_v
            
        tb = time.perf_counter()
        t_fan += (tb - ta)
        
        tc = time.perf_counter()
        uv_key = _island_uv_key(island_faces, uv_layer)
        isle.uv_key = uv_key
        
        isle.local_key = (len(f_tris), round(surf_area, 5), len(uv_key), round(uv_area, 5))
        ref_a = f_tris[0][0] if f_tris else (0.0, 0.0)
        ref_b = ref_a
        for t in f_tris:
            for v in t:
                if (v[0]-ref_a[0])**2 + (v[1]-ref_a[1])**2 > 1e-8:
                    ref_b = v
                    break
            if ref_b != ref_a:
                break
        isle.ref_a = ref_a
        isle.ref_b = ref_b
            
        td = time.perf_counter()
        t_key += (td - tc)
        
        isle.boundary_segs = _extract_boundary_segs(
            island_faces, face_index_set, uv_layer, uv_adj
        )
        te = time.perf_counter()
        t_bound += (te - td)
        
        islands.append(isle)

    utils_mod.log("timing_extract", f"fan: {t_fan*1000:.1f}ms, key: {t_key*1000:.1f}ms, bound: {t_bound*1000:.1f}ms")
    return islands


def _fan_tris_and_data(faces, uv_layer, matrix_world):
    tris = []
    jacobians = []
    face_normals = []
    face_areas = []
    grad_u_list = []
    grad_v_list = []
    total_uv_area = 0.0
    total_surf_area = 0.0

    identity_j = (1.0, 0.0, 0.0, 1.0)
    has_matrix = matrix_world is not None

    for face in faces:
        loops = face.loops
        if len(loops) < 3:
            continue
        l0 = loops[0]
        uv0 = l0[uv_layer].uv
        if has_matrix:
            p0 = matrix_world @ l0.vert.co
        else:
            p0 = l0.vert.co

        for i in range(1, len(loops) - 1):
            l1 = loops[i]
            l2 = loops[i + 1]
            uv1 = l1[uv_layer].uv
            uv2 = l2[uv_layer].uv
            if has_matrix:
                p1 = matrix_world @ l1.vert.co
                p2 = matrix_world @ l2.vert.co
            else:
                p1 = l1.vert.co
                p2 = l2.vert.co


            tris.append(((uv0.x, uv0.y), (uv1.x, uv1.y), (uv2.x, uv2.y)))


            eu = uv1 - uv0
            ev = uv2 - uv0
            det_uv = eu.x * ev.y - eu.y * ev.x
            uv_area = abs(det_uv) * 0.5
            total_uv_area += uv_area


            dp1 = p1 - p0
            dp2 = p2 - p0
            surf_area = dp1.cross(dp2).length * 0.5
            total_surf_area += surf_area
            
            lp0 = l0.vert.co
            lp1 = l1.vert.co
            lp2 = l2.vert.co
            ldp1 = lp1 - lp0
            ldp2 = lp2 - lp0
            cross_local = ldp1.cross(ldp2)
            face_normals.append((cross_local.x, cross_local.y, cross_local.z))
            face_areas.append(cross_local.length * 0.5)


            if abs(det_uv) < 1e-12:
                jacobians.append(identity_j)
                grad_u_list.append((0.0, 0.0, 0.0))
                grad_v_list.append((0.0, 0.0, 0.0))
                continue

            inv_det = 1.0 / det_uv
            Tu = (dp1 * ev.y - dp2 * eu.y) * inv_det
            Tv = (dp2 * eu.x - dp1 * ev.x) * inv_det

            E = Tu.dot(Tu)
            F = Tu.dot(Tv)
            G = Tv.dot(Tv)

            D = E * G - F * F
            if D < 1e-12:
                jacobians.append(identity_j)
                grad_u_list.append((0.0, 0.0, 0.0))
                grad_v_list.append((0.0, 0.0, 0.0))
                continue

            s = math.sqrt(D)
            t_sq = E + G + 2 * s
            if t_sq < 1e-12:
                jacobians.append(identity_j)
                grad_u_list.append((0.0, 0.0, 0.0))
                grad_v_list.append((0.0, 0.0, 0.0))
                continue

            t = math.sqrt(t_sq)
            M00 = (E + s) / t
            M01 = F / t
            M10 = F / t
            M11 = (G + s) / t
            jacobians.append((M00, M01, M10, M11))
            
            # Local gradients for UV axis projection
            # Tu and Tv here are 3D tangents mapping UV -> 3D.
            # We want grad_u and grad_v which are the dual basis vectors in the face plane.
            N_unscaled = Tu.cross(Tv)
            det_3d = Tu.dot(Tv.cross(N_unscaled))
            if abs(det_3d) > 1e-12:
                gu = Tv.cross(N_unscaled) / det_3d
                gv = N_unscaled.cross(Tu) / det_3d
                grad_u_list.append((gu.x, gu.y, gu.z))
                grad_v_list.append((gv.x, gv.y, gv.z))
            else:
                grad_u_list.append((0.0, 0.0, 0.0))
                grad_v_list.append((0.0, 0.0, 0.0))

    return tris, jacobians, total_uv_area, total_surf_area, face_normals, face_areas, grad_u_list, grad_v_list


def _island_uv_key(faces, uv_layer):
    uvs = set()
    for face in faces:
        for loop in face.loops:
            u = round(loop[uv_layer].uv.x, UV_DECIMAL)
            v = round(loop[uv_layer].uv.y, UV_DECIMAL)
            uvs.add((u, v))
    return frozenset(uvs)


def _aabb_overlap(a, b):
    return not (
        a[2] < b[0] - EPSILON or b[2] < a[0] - EPSILON or
        a[3] < b[1] - EPSILON or b[3] < a[1] - EPSILON
    )


def _aabb_identical(a, b):
    return (abs(a[0]-b[0]) < UV_EPS and abs(a[1]-b[1]) < UV_EPS and
            abs(a[2]-b[2]) < UV_EPS and abs(a[3]-b[3]) < UV_EPS)


def _seg_cross(p, r, q, s):
    # Strict interior only — endpoints excluded.
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < 1e-12:
        return False
    qp = (q[0]-p[0], q[1]-p[1])
    t  = (qp[0]*s[1] - qp[1]*s[0]) / rxs
    u  = (qp[0]*r[1] - qp[1]*r[0]) / rxs
    return EPSILON < t < 1.0 - EPSILON and EPSILON < u < 1.0 - EPSILON


def _segments_intersect(a1, a2, b1, b2):
    r = (a2[0]-a1[0], a2[1]-a1[1])
    s = (b2[0]-b1[0], b2[1]-b1[1])
    return _seg_cross(a1, r, b1, s)


def _boundaries_intersect(segs_a, segs_b):
    for a1, a2 in segs_a:
        for b1, b2 in segs_b:
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _tris_overlap_sat(t1, t2):
    def axes(tri):
        return [(-(tri[(i+1)%3][1]-tri[i][1]),
                   tri[(i+1)%3][0]-tri[i][0]) for i in range(3)]
    def project(tri, ax):
        dots = [v[0]*ax[0]+v[1]*ax[1] for v in tri]
        return min(dots), max(dots)
    for ax in axes(t1) + axes(t2):
        if abs(ax[0]) < 1e-10 and abs(ax[1]) < 1e-10:
            continue
        mn1, mx1 = project(t1, ax)
        mn2, mx2 = project(t2, ax)
        if mx1 < mn2 - EPSILON or mx2 < mn1 - EPSILON:
            return False
    return True


_SAT_SORT_MIN = 16  # proximity sort only pays off above this triangle count


def _point_in_polygon(pt, boundary_segs):
    x, y = pt
    inside = False
    for (p1, p2) in boundary_segs:
        x1, y1 = p1
        x2, y2 = p2
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _sat_overlap(island_a, island_b):
    """SAT fallback — handles containment and parallel-edge cases. Optimized with a spatial grid for large islands."""
    tris_a = island_a.tris
    tris_b = island_b.tris

    if len(tris_a) < _SAT_SORT_MIN or len(tris_b) < _SAT_SORT_MIN:
        for ta in tris_a:
            for tb in tris_b:
                if _tris_overlap_sat(ta, tb):
                    return True
        return False

    mn_u, mn_v, mx_u, mx_v = island_b.aabb
    cell_size = max(0.01, (mx_u - mn_u) / 10.0, (mx_v - mn_v) / 10.0)
    
    grid_b = {}
    for ib, tb in enumerate(tris_b):
        b_mn_u = min(tb[0][0], tb[1][0], tb[2][0])
        b_mn_v = min(tb[0][1], tb[1][1], tb[2][1])
        b_mx_u = max(tb[0][0], tb[1][0], tb[2][0])
        b_mx_v = max(tb[0][1], tb[1][1], tb[2][1])
        
        cx0 = int(math.floor(b_mn_u / cell_size))
        cy0 = int(math.floor(b_mn_v / cell_size))
        cx1 = int(math.floor(b_mx_u / cell_size))
        cy1 = int(math.floor(b_mx_v / cell_size))
        
        entry = (ib, b_mn_u, b_mn_v, b_mx_u, b_mx_v)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                key = (cx, cy)
                if key not in grid_b:
                    grid_b[key] = []
                grid_b[key].append(entry)

    for ta in tris_a:
        a_mn_u = min(ta[0][0], ta[1][0], ta[2][0])
        a_mn_v = min(ta[0][1], ta[1][1], ta[2][1])
        a_mx_u = max(ta[0][0], ta[1][0], ta[2][0])
        a_mx_v = max(ta[0][1], ta[1][1], ta[2][1])
        
        cx0 = int(math.floor(a_mn_u / cell_size))
        cy0 = int(math.floor(a_mn_v / cell_size))
        cx1 = int(math.floor(a_mx_u / cell_size))
        cy1 = int(math.floor(a_mx_v / cell_size))
        
        tested_b = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                key = (cx, cy)
                if key in grid_b:
                    for ib, b_mn_u, b_mn_v, b_mx_u, b_mx_v in grid_b[key]:
                        if ib in tested_b:
                            continue
                        tested_b.add(ib)
                        if not (a_mx_u < b_mn_u - EPSILON or b_mx_u < a_mn_u - EPSILON or
                                a_mx_v < b_mn_v - EPSILON or b_mx_v < a_mn_v - EPSILON):
                            if _tris_overlap_sat(ta, tris_b[ib]):
                                return True
    return False


def _islands_overlap_contour(a, b):
    # Stage 1: boundary crossing.
    if a.boundary_segs and b.boundary_segs:
        if _boundaries_intersect(a.boundary_segs, b.boundary_segs):
            return True
            
        # Stage 2: Fast point-in-polygon check for containment.
        if a.tri_centers and _point_in_polygon(a.tri_centers[0], b.boundary_segs):
            return True
        if b.tri_centers and _point_in_polygon(b.tri_centers[0], a.boundary_segs):
            return True
            
    # Stage 3: SAT — handles containment and parallel-edge cases.
    return _sat_overlap(a, b)


def find_tile_crossing_islands(islands):
    """Return indices of islands whose AABB spans an integer UV tile boundary."""
    crossing = set()
    for i, isle in enumerate(islands):
        mn_u, mn_v, mx_u, mx_v = isle.aabb
        if (math.floor(mn_u + UV_EPS) != math.floor(mx_u - UV_EPS) or
                math.floor(mn_v + UV_EPS) != math.floor(mx_v - UV_EPS)):
            crossing.add(i)
    return crossing


def _find_stacked(islands):
    """Return (stacked_idx, stacked_pairs) for islands with identical UV positions."""
    stacked = set()
    pairs   = set()
    n       = len(islands)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = islands[i], islands[j]
            if (_aabb_identical(a.aabb, b.aabb)
                    and a.uv_key and b.uv_key
                    and a.uv_key == b.uv_key):
                stacked.add(i)
                stacked.add(j)
                pairs.add((i, j))
    return frozenset(stacked), pairs


def _build_spatial_grid(islands, cell_size):
    grid = {}
    for isle in islands:
        mn_u, mn_v, mx_u, mx_v = isle.aabb
        cx0 = int(math.floor(mn_u / cell_size))
        cy0 = int(math.floor(mn_v / cell_size))
        cx1 = int(math.floor(mx_u / cell_size))
        cy1 = int(math.floor(mx_v / cell_size))
        for cx in range(cx0, cx1+1):
            for cy in range(cy0, cy1+1):
                key = (cx, cy)
                if key not in grid:
                    grid[key] = []
                grid[key].append(isle)
    return grid


def _get_overlapping_pairs(islands, stacked_idx, stacked_pairs):
    """Spatial grid cull → boundary crossing → SAT. Skips confirmed stacked pairs."""
    if not islands:
        return set()

    isle_to_idx = {id(isle): i for i, isle in enumerate(islands)}
    diags = sorted(
        math.sqrt((i.aabb[2]-i.aabb[0])**2 + (i.aabb[3]-i.aabb[1])**2)
        for i in islands
    )
    median_diag = diags[len(diags)//2] if diags else 0.1
    cell_size   = max(0.05, min(0.5, median_diag * 2.0))
    grid        = _build_spatial_grid(islands, cell_size)

    island_pairs = set()
    tested       = set()

    for cell_isles in grid.values():
        if len(cell_isles) < 2:
            continue
        n = len(cell_isles)
        for i in range(n):
            for j in range(i+1, n):
                a  = cell_isles[i]
                b  = cell_isles[j]
                ia = isle_to_idx[id(a)]
                ib = isle_to_idx[id(b)]
                pk = (ia, ib) if ia < ib else (ib, ia)
                if pk in tested:
                    continue
                tested.add(pk)
                if pk in stacked_pairs:
                    continue
                if not _aabb_overlap(a.aabb, b.aabb):
                    continue
                if _islands_overlap_contour(a, b):
                    island_pairs.add(pk)

    return island_pairs


def _get_overlapping_pairs_cross(islands_a, islands_b):
    """Cross-object overlap: a vs b only, spatial grid on b set."""
    inter_a, inter_b = set(), set()
    stack_a, stack_b = set(), set()
    pairs = []

    if not islands_a or not islands_b:
        return inter_a, inter_b, stack_a, stack_b, pairs

    idx_b = {id(isle): i for i, isle in enumerate(islands_b)}
    diags = sorted(
        math.sqrt((i.aabb[2]-i.aabb[0])**2 + (i.aabb[3]-i.aabb[1])**2)
        for i in islands_b
    )
    median_diag = diags[len(diags)//2] if diags else 0.1
    cell_size   = max(0.05, min(0.5, median_diag * 2.0))
    grid_b      = _build_spatial_grid(islands_b, cell_size)

    tested = set()
    for ia, a in enumerate(islands_a):
        mn_u, mn_v, mx_u, mx_v = a.aabb
        cx0 = int(math.floor(mn_u / cell_size))
        cy0 = int(math.floor(mn_v / cell_size))
        cx1 = int(math.floor(mx_u / cell_size))
        cy1 = int(math.floor(mx_v / cell_size))
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for b in grid_b.get((cx, cy), ()):
                    ib = idx_b[id(b)]
                    pk = (ia, ib)
                    if pk in tested:
                        continue
                    tested.add(pk)
                    if not _aabb_overlap(a.aabb, b.aabb):
                        continue
                    if (_aabb_identical(a.aabb, b.aabb)
                            and a.uv_key and b.uv_key
                            and a.uv_key == b.uv_key):
                        stack_a.add(ia)
                        stack_b.add(ib)
                        continue
                    if _islands_overlap_contour(a, b):
                        inter_a.add(ia)
                        inter_b.add(ib)
                        pairs.append((ia, ib))
    return inter_a, inter_b, stack_a, stack_b, pairs



def _get_overlapping_pairs_cached(islands, stacked_idx, stacked_pairs,
                                   changed_keys, prev_pair_cache):
    """Spatial grid cull -> overlap test with per-pair result caching."""
    if not islands:
        return set(), {}

    isle_to_idx = {id(isle): i for i, isle in enumerate(islands)}
    diags = sorted(
        math.sqrt((i.aabb[2]-i.aabb[0])**2 + (i.aabb[3]-i.aabb[1])**2)
        for i in islands
    )
    median_diag = diags[len(diags)//2] if diags else 0.1
    cell_size   = max(0.05, min(0.5, median_diag * 2.0))
    grid        = _build_spatial_grid(islands, cell_size)

    island_pairs = set()
    new_cache    = {}
    tested       = set()

    for cell_isles in grid.values():
        if len(cell_isles) < 2:
            continue
        n = len(cell_isles)
        for i in range(n):
            for j in range(i + 1, n):
                a  = cell_isles[i]
                b  = cell_isles[j]
                ia = isle_to_idx[id(a)]
                ib = isle_to_idx[id(b)]
                pk = (ia, ib) if ia < ib else (ib, ia)
                if pk in tested:
                    continue
                tested.add(pk)
                if pk in stacked_pairs:
                    continue
                if not _aabb_overlap(a.aabb, b.aabb):
                    continue

                ka, kb = a.uv_key, b.uv_key
                # Stable ordering by frozenset hash.
                ck = (ka, kb) if hash(ka) <= hash(kb) else (kb, ka)

                if (ka not in changed_keys and kb not in changed_keys
                        and prev_pair_cache is not None
                        and ck in prev_pair_cache):
                    overlaps = prev_pair_cache[ck]
                else:
                    overlaps = _islands_overlap_contour(a, b)

                new_cache[ck] = overlaps
                if overlaps:
                    island_pairs.add(pk)

    return island_pairs, new_cache


def _get_overlapping_pairs_cross_cached(islands_a, islands_b,
                                         changed_keys_a, changed_keys_b,
                                         prev_pair_cache):
    """Cross-object overlap with per-pair caching."""
    inter_a, inter_b = set(), set()
    stack_a, stack_b = set(), set()
    pairs    = []
    new_cache = {}

    if not islands_a or not islands_b:
        return inter_a, inter_b, stack_a, stack_b, pairs, new_cache

    idx_b = {id(isle): i for i, isle in enumerate(islands_b)}
    diags = sorted(
        math.sqrt((i.aabb[2]-i.aabb[0])**2 + (i.aabb[3]-i.aabb[1])**2)
        for i in islands_b
    )
    median_diag = diags[len(diags)//2] if diags else 0.1
    cell_size   = max(0.05, min(0.5, median_diag * 2.0))
    grid_b      = _build_spatial_grid(islands_b, cell_size)

    tested = set()
    for ia, a in enumerate(islands_a):
        mn_u, mn_v, mx_u, mx_v = a.aabb
        cx0 = int(math.floor(mn_u / cell_size))
        cy0 = int(math.floor(mn_v / cell_size))
        cx1 = int(math.floor(mx_u / cell_size))
        cy1 = int(math.floor(mx_v / cell_size))
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for b in grid_b.get((cx, cy), ()):
                    ib = idx_b[id(b)]
                    pk = (ia, ib)
                    if pk in tested:
                        continue
                    tested.add(pk)
                    if not _aabb_overlap(a.aabb, b.aabb):
                        continue
                    if (_aabb_identical(a.aabb, b.aabb)
                            and a.uv_key and b.uv_key
                            and a.uv_key == b.uv_key):
                        stack_a.add(ia)
                        stack_b.add(ib)
                        continue

                    ka, kb = a.uv_key, b.uv_key
                    ck = (ka, kb)

                    if (ka not in changed_keys_a and kb not in changed_keys_b
                            and prev_pair_cache is not None
                            and ck in prev_pair_cache):
                        overlaps = prev_pair_cache[ck]
                    else:
                        overlaps = _islands_overlap_contour(a, b)

                    new_cache[ck] = overlaps
                    if overlaps:
                        inter_a.add(ia)
                        inter_b.add(ib)
                        pairs.append((ia, ib))

    return inter_a, inter_b, stack_a, stack_b, pairs, new_cache

def classify_islands(islands, prev_inter_idx=None, prev_stack_idx=None,
                     prev_uv_key_hash=None, prev_inter_pairs=None,
                     prev_island_keys=None, prev_pair_cache=None):
    """Returns (inter_idx, stack_idx, uv_key_hash, inter_pairs,
                island_keys, pair_cache).

    island_keys: list of uv_key per island; pair_cache: per-pair overlap results."""
    if not islands:
        return frozenset(), frozenset(), 0, frozenset(), [], {}

    cur_island_keys = [isle.uv_key for isle in islands]
    cur_uv_key_hash = hash(frozenset(
        (i, k) for i, k in enumerate(cur_island_keys) if k is not None
    ))


    if (prev_uv_key_hash == cur_uv_key_hash
            and prev_inter_idx is not None
            and prev_stack_idx is not None):
        utils.log("classify", "cache hit")
        return (prev_inter_idx, prev_stack_idx, cur_uv_key_hash,
                prev_inter_pairs or frozenset(),
                prev_island_keys or cur_island_keys,
                prev_pair_cache or {})


    if prev_island_keys is not None:
        prev_counts = _Counter(k for k in prev_island_keys if k is not None)
        cur_counts  = _Counter(k for k in cur_island_keys  if k is not None)
        changed_keys = set()
        for k in set(prev_counts) | set(cur_counts):
            if prev_counts[k] != cur_counts[k]:
                changed_keys.add(k)
        if None in cur_island_keys:
            changed_keys.add(None)
    else:
        changed_keys = set(cur_island_keys)

    stack_idx, stacked_pairs = _find_stacked(islands)
    island_pairs, new_pair_cache = _get_overlapping_pairs_cached(
        islands, stack_idx, stacked_pairs, changed_keys, prev_pair_cache
    )
    inter_idx = frozenset(idx for pk in island_pairs for idx in pk)

    _reused = sum(1 for ck in new_pair_cache
                  if prev_pair_cache and ck in prev_pair_cache)
    utils.log("classify", (
        f"{len(islands)} islands, inter={sorted(inter_idx)}, "
        f"stack={sorted(stack_idx)}, pairs={sorted(island_pairs)}, "
        f"pair_cache reused={_reused}/{len(new_pair_cache)}"
    ))

    return inter_idx, stack_idx, cur_uv_key_hash, island_pairs, cur_island_keys, new_pair_cache


def classify_islands_cross(islands_a, islands_b,
                            prev_inter_a=None, prev_inter_b=None,
                            prev_stack_a=None, prev_stack_b=None,
                            prev_uv_hash=None, prev_inter_pairs=None,
                            prev_island_keys_a=None, prev_island_keys_b=None,
                            prev_pair_cache=None):
    """Cross-object. Returns (inter_a, inter_b, stack_a, stack_b, uv_hash,
                              inter_pairs, island_keys_a, island_keys_b, pair_cache)."""
    if not islands_a or not islands_b:
        return frozenset(), frozenset(), frozenset(), frozenset(), 0, [], [], [], {}

    cur_keys_a = [isle.uv_key for isle in islands_a]
    cur_keys_b = [isle.uv_key for isle in islands_b]
    cur_uv_hash = hash((
        frozenset((i, k) for i, k in enumerate(cur_keys_a) if k is not None),
        frozenset((i, k) for i, k in enumerate(cur_keys_b) if k is not None),
    ))


    if prev_inter_a is not None and prev_uv_hash == cur_uv_hash:
        return (prev_inter_a, prev_inter_b,
                prev_stack_a or frozenset(), prev_stack_b or frozenset(),
                cur_uv_hash, prev_inter_pairs or [],
                prev_island_keys_a or cur_keys_a,
                prev_island_keys_b or cur_keys_b,
                prev_pair_cache or {})

    def _changed(prev_keys, cur_keys):
        if prev_keys is None:
            return set(cur_keys)
        pc = _Counter(k for k in prev_keys if k is not None)
        cc = _Counter(k for k in cur_keys  if k is not None)
        diff = set()
        for k in set(pc) | set(cc):
            if pc[k] != cc[k]:
                diff.add(k)
        if None in cur_keys:
            diff.add(None)
        return diff

    changed_a = _changed(prev_island_keys_a, cur_keys_a)
    changed_b = _changed(prev_island_keys_b, cur_keys_b)

    raw_a, raw_b, s_a, s_b, pairs, new_pair_cache = _get_overlapping_pairs_cross_cached(
        islands_a, islands_b, changed_a, changed_b, prev_pair_cache
    )

    _reused = sum(1 for ck in new_pair_cache
                  if prev_pair_cache and ck in prev_pair_cache)
    utils.log("classify_cross", (
        f"A:{len(islands_a)} B:{len(islands_b)} islands, "
        f"inter_a={sorted(raw_a)}, inter_b={sorted(raw_b)}, "
        f"stack_a={sorted(s_a)}, stack_b={sorted(s_b)}, "
        f"pairs={pairs}, pair_cache reused={_reused}/{len(new_pair_cache)}"
    ))

    return (frozenset(raw_a), frozenset(raw_b), frozenset(s_a), frozenset(s_b),
            cur_uv_hash, pairs, cur_keys_a, cur_keys_b, new_pair_cache)


def generate_hatch(tris, gap=0.01, angle_deg=45):
    if not tris:
        return []
    rad   = math.radians(angle_deg)
    cos_a = math.cos(-rad)
    sin_a = math.sin(-rad)

    def rot(p):
        return (p[0]*cos_a - p[1]*sin_a, p[0]*sin_a + p[1]*cos_a)
    def unrot(p):
        return (p[0]*cos_a + p[1]*sin_a, -p[0]*sin_a + p[1]*cos_a)

    rot_tris = [[rot(v) for v in tri] for tri in tris]
    all_pts  = [v for tri in rot_tris for v in tri]
    min_v    = min(p[1] for p in all_pts)
    max_v    = max(p[1] for p in all_pts)

    segments = []
    y = min_v + gap * 0.5
    while y <= max_v + EPSILON:
        xs = []
        for tri in rot_tris:
            for k in range(3):
                v1, v2 = tri[k], tri[(k+1) % 3]
                if (v1[1] <= y < v2[1]) or (v2[1] <= y < v1[1]):
                    denom = v2[1] - v1[1]
                    if abs(denom) > 1e-10:
                        t = (y - v1[1]) / denom
                        xs.append(v1[0] + t*(v2[0]-v1[0]))
        xs.sort()
        # Odd x count: merge near-equal clusters.
        if len(xs) % 2 == 1:
            merged = []
            i = 0
            while i < len(xs):
                j = i + 1
                while j < len(xs) and abs(xs[j] - xs[i]) < 1e-7:
                    j += 1
                if (j - i) % 2 == 1:
                    merged.append(xs[i])
                i = j
            xs = merged
        for k in range(0, len(xs) - 1, 2):
            p1 = unrot((xs[k],   y))
            p2 = unrot((xs[k+1], y))
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            if dx*dx + dy*dy > EPSILON:
                segments.append((p1, p2))
        y += gap
    return segments


def generate_cross_hatch(tris, gap=0.01, angle_deg=45):
    return (generate_hatch(tris, gap=gap, angle_deg= angle_deg) +
            generate_hatch(tris, gap=gap, angle_deg=-angle_deg))


def normalize_island(isle):
    """Translate island so its centroid sits in tile (0,0). Recomputes uv_key with modulo."""
    if not isle.tris:
        norm               = Island([], isle.color, isle.object_name)
        norm.boundary_segs = []
        norm.uv_key        = isle.uv_key
        return norm

    cx = (isle.aabb[0] + isle.aabb[2]) * 0.5
    cy = (isle.aabb[1] + isle.aabb[3]) * 0.5
    du = max(0, math.floor(cx))
    dv = max(0, math.floor(cy))

    translated_tris = [
        tuple((u - du, v - dv) for u, v in tri)
        for tri in isle.tris
    ]
    norm               = Island(translated_tris, isle.color, isle.object_name)
    norm.boundary_segs = []
    norm.uv_key        = (
        frozenset(
            (round(u % 1.0, UV_DECIMAL), round(v % 1.0, UV_DECIMAL))
            for u, v in isle.uv_key
        )
        if isle.uv_key else None
    )
    return norm