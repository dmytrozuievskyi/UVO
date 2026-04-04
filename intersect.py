import math

EPSILON    = 1e-6
UV_EPS     = 1e-4
UV_DECIMAL = 3


class Island:
    __slots__ = ('tris', 'aabb', 'uv_key', 'color', 'object_name',
                 'boundary_segs', 'tri_centers')

    def __init__(self, tris, color, object_name=''):
        self.tris          = tris
        self.color         = color
        self.object_name   = object_name
        self.uv_key        = None
        self.boundary_segs = []

        if tris:
            all_u = [v[0] for t in tris for v in t]
            all_v = [v[1] for t in tris for v in t]
            self.aabb = (min(all_u), min(all_v), max(all_u), max(all_v))
            # Precomputed centroids for proximity sorting in _sat_overlap.
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
    """One O(total_loops) pass to build face-edge→loop map, then O(1) lookups per edge."""
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
    """Collect UV edges forming the outer contour of the island.

    An edge is a boundary if its neighbour is outside the island or
    not UV-adjacent (i.e. there is a UV seam between them).
    """
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
                    object_name=''):
    bm_copy.faces.ensure_lookup_table()
    uv_adj = _build_uv_adjacency(bm_copy, uv_layer)

    # Pass 1: flood-fill to get total island count before colouring,
    # so hue wheel slots can be divided equally upfront.
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

    # Pass 2: build Island objects and assign colours.
    islands = []
    for idx, face_index_set in enumerate(face_groups):
        col = utils_mod.get_distinct_color(
            idx, total, seed_offset=obj_seed, alpha=alpha_val
        )

        island_faces = [bm_copy.faces[i] for i in face_index_set]
        tris = _fan_tris_from_faces(island_faces, uv_layer)

        if tris:
            isle               = Island(tris, col, object_name)
            isle.uv_key        = _island_uv_key(island_faces, uv_layer)
            isle.boundary_segs = _extract_boundary_segs(
                island_faces, face_index_set, uv_layer, uv_adj
            )
            islands.append(isle)

    return islands


def _fan_tris_from_faces(faces, uv_layer):
    tris = []
    for face in faces:
        loops = face.loops
        if len(loops) < 3:
            continue
        uv0 = (loops[0][uv_layer].uv.x, loops[0][uv_layer].uv.y)
        for i in range(1, len(loops) - 1):
            uv1 = (loops[i][uv_layer].uv.x,   loops[i][uv_layer].uv.y)
            uv2 = (loops[i+1][uv_layer].uv.x, loops[i+1][uv_layer].uv.y)
            tris.append((uv0, uv1, uv2))
    return tris


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
    # Strict interior only — endpoints excluded to avoid false positives
    # where a vertex merely touches an adjacent edge.
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


def _sat_overlap(island_a, island_b):
    """SAT fallback — handles containment and parallel-edge cases that boundary
    crossing misses. Proximity-sorted for early exit on large islands."""
    tris_a = island_a.tris
    tris_b = island_b.tris

    if len(tris_a) >= _SAT_SORT_MIN and len(tris_b) >= _SAT_SORT_MIN:
        bx = (island_b.aabb[0] + island_b.aabb[2]) * 0.5
        by = (island_b.aabb[1] + island_b.aabb[3]) * 0.5
        ax = (island_a.aabb[0] + island_a.aabb[2]) * 0.5
        ay = (island_a.aabb[1] + island_a.aabb[3]) * 0.5
        ctrs_a = island_a.tri_centers
        ctrs_b = island_b.tri_centers
        order_a = sorted(range(len(tris_a)),
                         key=lambda i: (ctrs_a[i][0]-bx)**2 + (ctrs_a[i][1]-by)**2)
        order_b = sorted(range(len(tris_b)),
                         key=lambda i: (ctrs_b[i][0]-ax)**2 + (ctrs_b[i][1]-ay)**2)
        for ia in order_a:
            for ib in order_b:
                if _tris_overlap_sat(tris_a[ia], tris_b[ib]):
                    return True
        return False

    for ta in tris_a:
        for tb in tris_b:
            if _tris_overlap_sat(ta, tb):
                return True
    return False


def _islands_overlap_contour(a, b):
    # Stage 1: boundary crossing — fast, handles most partial overlaps.
    # Falls through to SAT for closed meshes with no boundary_segs.
    if a.boundary_segs and b.boundary_segs:
        if _boundaries_intersect(a.boundary_segs, b.boundary_segs):
            return True
    # Stage 2: SAT — handles containment and parallel-edge cases.
    return _sat_overlap(a, b)


def find_tile_crossing_islands(islands):
    """Return indices of islands whose AABB spans an integer UV tile boundary.

    Islands flush with a tile edge (e.g. max_u == 1.0) are not flagged.
    """
    crossing = set()
    for i, isle in enumerate(islands):
        mn_u, mn_v, mx_u, mx_v = isle.aabb
        if (math.floor(mn_u + UV_EPS) != math.floor(mx_u - UV_EPS) or
                math.floor(mn_v + UV_EPS) != math.floor(mx_v - UV_EPS)):
            crossing.add(i)
    return crossing


def _find_stacked(islands):
    """Return (stacked_idx, stacked_pairs) for islands sharing identical UV positions.

    stacked_idx: frozenset of all indices in at least one stacked pair.
    stacked_pairs: set of exact (ia, ib) pairs.
    """
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
    """Spatial grid cull -> overlap test, with per-pair result caching.

    Only re-tests pairs where at least one island uv_key is in changed_keys.
    Uses hash() for stable pair-key ordering across rebuilds (not id()).
    """
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
                # Stable ordering by hash() — consistent across rebuilds
                # because frozenset hash depends only on content, not address.
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
    """Cross-object overlap with per-pair caching.

    Cache key is (ka, kb) — directional (A->B), no reordering needed.
    Frozenset equality is by value so lookup is stable across rebuilds.
    """
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
                    ck = (ka, kb)   # directional: A->B always

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

    island_keys: list of uv_key per island (for diffing next call).
    pair_cache:  per-pair overlap results for selective re-test.
    """
    if not islands:
        return frozenset(), frozenset(), 0, frozenset(), [], {}

    cur_island_keys = [isle.uv_key for isle in islands]
    cur_uv_key_hash = hash(frozenset(
        (i, k) for i, k in enumerate(cur_island_keys) if k is not None
    ))

    # Full cache hit: nothing changed.
    if (prev_uv_key_hash == cur_uv_key_hash
            and prev_inter_idx is not None
            and prev_stack_idx is not None):
        try:
            from . import utils
        except ImportError:
            import utils
        utils.log("classify", "cache hit")
        return (prev_inter_idx, prev_stack_idx, cur_uv_key_hash,
                prev_inter_pairs or frozenset(),
                prev_island_keys or cur_island_keys,
                prev_pair_cache or {})

    # Diff old vs new island keys to find what moved.
    from collections import Counter as _Counter
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

    try:
        from . import utils
    except ImportError:
        import utils
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

    # Full cache hit.
    if prev_inter_a is not None and prev_uv_hash == cur_uv_hash:
        return (prev_inter_a, prev_inter_b,
                prev_stack_a or frozenset(), prev_stack_b or frozenset(),
                cur_uv_hash, prev_inter_pairs or [],
                prev_island_keys_a or cur_keys_a,
                prev_island_keys_b or cur_keys_b,
                prev_pair_cache or {})

    from collections import Counter as _Counter
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

    try:
        from . import utils
    except ImportError:
        import utils
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
        # Odd x count means a scanline hit a shared vertex — merge near-equal
        # clusters: odd-sized = local extremum (keep one); even-sized = discard.
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
    """Return a copy of isle rigidly translated so its centroid sits in tile (0,0).

    Rigid translation preserves triangle shapes. uv_key is recomputed with
    per-vertex modulo so stacked detection works across tiles.
    """
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