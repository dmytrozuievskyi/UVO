import bpy
import bmesh
import gpu
import math
import struct
import time
import traceback
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
from . import utils
from . import intersect as ix
from . import offscreen
from . import padding
from . import stretch

draw_handler   = None
is_calculating = False

# {obj_name: {'hash': int, 'id_batch': batch|None, 'islands': [Island]}}
_obj_cache = {}

_isect_self_cache  = {}   # per-object intersect results
_isect_cross_cache = {}   # per-object-pair cross-object intersect results

_intersect_batches = {'hatch': None, 'checker': None}

_hatch_seg_cache       = {}   # for global_inter
_cross_hatch_seg_cache = {}   # for global_stack

_inter_island_tris = []   # tris fed to offscreen.render() each frame

# gray = 1/(n+1), threshold = gray*1.5 — any pixel covered by 2+ islands → red fill.
_inter_gray      = 0.5
_inter_threshold = 0.6

_shader = None

# Async classify tracking
_classify_job_id    = 0   # id of the most recently dispatched classify job
_classify_timer_fn  = None  # bpy timer reference

# Sentinel: pre-pass failed for this object — use n=1 fallback, keeping
# palette offsets consistent with what the pre-pass recorded.
_PREPASS_FAILED = object()

_DEBOUNCE_DELAY = 0.5   # seconds; rebuild fires once the user stops editing
_debounce_fn    = None


def _schedule_debounce():
    global _debounce_fn
    _cancel_debounce()

    def _fire():
        global _debounce_fn
        _debounce_fn = None
        if not is_calculating:
            update_batches_safe(bpy.context)
            try:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            area.tag_redraw()
            except Exception:
                pass
        return None  # unregister

    _debounce_fn = _fire
    bpy.app.timers.register(_fire, first_interval=_DEBOUNCE_DELAY)


def _cancel_debounce():
    global _debounce_fn
    if _debounce_fn is not None:
        try:
            bpy.app.timers.unregister(_debounce_fn)
        except Exception:
            pass
        _debounce_fn = None


def full_refresh(context):
    """Clear all caches and rebuild. Call when settings change (opacity, mode, etc.)."""
    _cancel_classify_timer()
    _obj_cache.clear()
    _isect_self_cache.clear()
    _isect_cross_cache.clear()
    _hatch_seg_cache.clear()
    _cross_hatch_seg_cache.clear()
    update_batches_safe(context)
    props = context.scene.uv_id_props
    if props.show_padding and not props.is_muted:
        _rebuild_padding_batches(props)
    try:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
    except Exception:
        pass


def _get_shader():
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    return _shader


def _uv_hash(bm, uv_layer):
    # Polynomial rolling hash — order-sensitive so a rotated island hashes differently.
    _pack = struct.pack
    h = 0
    for face in bm.faces:
        for loop in face.loops:
            uv = loop[uv_layer].uv
            h = (h * 1000003 ^ hash(_pack('2f', uv.x, uv.y))) & 0xFFFFFFFFFFFFFFFF
    return h


def _mesh_connected_groups(bm):
    """Group faces by 3D edge connectivity, ignoring UV seams."""
    visited = set()
    groups  = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        group = set()
        stack = [seed]
        visited.add(seed.index)
        group.add(seed.index)
        while stack:
            curr = stack.pop()
            for edge in curr.edges:
                for lf in edge.link_faces:
                    if lf.index not in visited:
                        visited.add(lf.index)
                        group.add(lf.index)
                        stack.append(lf)
        groups.append(group)
    return groups


def _build_obj_data(obj, uv_id_mode, uv_id_alpha,
                    obj_index=0, total_objs=1,
                    group_offset=0, total_global_groups=1,
                    precomputed_groups=None):
    # Re-check mode — it can change between depsgraph event and this call.
    if obj.mode != 'EDIT':
        return None, None, None, None, None
    try:
        bm_live = bmesh.from_edit_mesh(obj.data)
        bm_copy = bm_live.copy()
    except Exception:
        return None, None, None, None, None

    current_hash = None
    try:
        bm_copy.faces.ensure_lookup_table()
        if len(bm_copy.faces) == 0:
            return None, None, None, None, None

        uv_layer     = bm_copy.loops.layers.uv.verify()
        current_hash = _uv_hash(bm_copy, uv_layer)

        cached = _obj_cache.get(obj.name)
        if cached and cached['hash'] == current_hash:
            utils.log("id_cache", f"{obj.name}: hit (hash={current_hash})")
            return current_hash, None, None, None, None

        obj_seed = utils.get_string_hash(obj.name)
        islands  = ix.extract_islands(
            bm_copy, uv_layer, uv_id_alpha, obj_seed, utils, obj.name, obj.matrix_world
        )
        utils.log("id_extract", (
            f"{obj.name}: {len(islands)} islands, "
            f"{sum(len(i.tris) for i in islands)} tris, "
            f"{sum(len(i.boundary_segs) for i in islands)} boundary_segs"
        ))

        coords, colors = [], []
        shader = _get_shader()

        if uv_id_mode == 'OBJECT':
            obj_col = utils.get_distinct_color(
                obj_index, total_objs, seed_offset=0.0, alpha=uv_id_alpha
            )
            for isle in islands:
                for tri in isle.tris:
                    for v in tri:
                        coords.append((v[0], v[1], 0.0))
                        colors.append(obj_col)
        else:
            # CONNECTED: one colour per 3D-connected piece, global palette so
            # all objects share maximally separated hues.
            if precomputed_groups is _PREPASS_FAILED:
                topo_groups = [set(f.index for f in bm_copy.faces)]
            elif precomputed_groups is not None:
                topo_groups = precomputed_groups
            else:
                topo_groups = _mesh_connected_groups(bm_copy)
            utils.log("connected", (
                f"{obj.name}: {len(topo_groups)} 3D-connected groups "
                f"from {len(bm_copy.faces)} faces "
                f"(global offset {group_offset}/{total_global_groups})"
            ))
            face_color = {}
            for gi, group in enumerate(topo_groups):
                col = utils.get_distinct_color(
                    group_offset + gi, total_global_groups,
                    seed_offset=0.0, alpha=uv_id_alpha
                )
                for fi in group:
                    face_color[fi] = col

            for face in bm_copy.faces:
                col = face_color.get(face.index)
                if col is None:
                    continue
                loops = face.loops
                if len(loops) < 3:
                    continue
                uv0 = loops[0][uv_layer].uv
                p0  = (uv0.x, uv0.y, 0.0)
                for i in range(1, len(loops) - 1):
                    uv1 = loops[i][uv_layer].uv
                    uv2 = loops[i + 1][uv_layer].uv
                    coords += [p0, (uv1.x, uv1.y, 0.0), (uv2.x, uv2.y, 0.0)]
                    colors += [col, col, col]

        id_batch = (
            batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})
            if coords else None
        )
        return current_hash, id_batch, islands, list(coords), list(colors)

    except Exception as e:
        utils.log("build", f"error ({obj.name}): {e}")
        traceback.print_exc()
        return current_hash, None, None, None, None
    finally:
        if bm_copy:
            bm_copy.free()



def _serialize_islands_for_worker(tiled):
    """Pack all cached island data into plain types for the worker subprocess.

    Also bundles the previous classify cache entries so the worker can do
    per-pair caching without re-testing unchanged pairs.
    """
    import sys as _sys
    pkg = _sys.modules.get(__package__)

    objects = []
    for name, cache in _obj_cache.items():
        islands = cache.get('islands') or []
        prev_self = _isect_self_cache.get(name, {})
        cur_hash = cache.get('hash')

        if pkg and pkg.get_synced_hash(name) == cur_hash:
            ser_islands = None
        else:
            ser_islands = []
            for isle in islands:
                # Flatten tris and boundary_segs to avoid nested tuple overhead
                flat_tris = [
                    (t[0][0], t[0][1], t[1][0], t[1][1], t[2][0], t[2][1])
                    for t in isle.tris
                ]
                flat_segs = [
                    (s[0][0], s[0][1], s[1][0], s[1][1])
                    for s in isle.boundary_segs
                ]
                ser_islands.append({
                    'flat_tris':   flat_tris,
                    'flat_segs':   flat_segs,
                    'color':       isle.color,
                    'object_name': isle.object_name,
                    'uv_key':      isle.uv_key,
                })
            if pkg:
                pkg.mark_synced(name, cur_hash)

        objects.append({
            'name':      name,
            'hash':      cur_hash,
            'islands':   ser_islands,
            'prev_self': {
                'inter_idx':   prev_self.get('inter_idx'),
                'stack_idx':   prev_self.get('stack_idx'),
                'uv_key_hash': prev_self.get('uv_key_hash'),
                'inter_pairs': prev_self.get('inter_pairs'),
                'island_keys': prev_self.get('island_keys'),
                'pair_cache':  prev_self.get('pair_cache'),
            },
        })

    # Bundle cross-cache previous state
    cross_prev = {}
    for pair_key, entry in _isect_cross_cache.items():
        cross_prev[pair_key] = {
            'inter_a':       entry.get('inter_a'),
            'inter_b':       entry.get('inter_b'),
            'stack_a':       entry.get('stack_a'),
            'stack_b':       entry.get('stack_b'),
            'uv_hash':       entry.get('uv_hash'),
            'inter_pairs':   entry.get('inter_pairs'),
            'island_keys_a': entry.get('island_keys_a'),
            'island_keys_b': entry.get('island_keys_b'),
            'pair_cache':    entry.get('pair_cache'),
        }

    return objects, cross_prev


def _dispatch_classify_job(props):
    """Serialize island data and send a classify_all job to the worker.

    Returns True if the job was dispatched, False if the worker is unavailable.
    """
    global _classify_job_id
    import sys as _sys
    pkg = _sys.modules.get(__package__)
    if pkg is None:
        return False

    tiled = (props.intersect_uv_mode == 'TILED')
    objects, cross_prev = _serialize_islands_for_worker(tiled)

    job_id = pkg.next_job_id()
    _classify_job_id = job_id

    ok = pkg.send_job({
        'id':         job_id,
        'type':       'classify_all',
        'tiled':      tiled,
        'objects':    objects,
        'cross_prev': cross_prev,
    })
    utils.log("async", f"classify job dispatched id={job_id} ok={ok}")
    return ok


def _start_classify_timer():
    """Register a repeating timer to poll for classify results."""
    global _classify_timer_fn
    _cancel_classify_timer()

    def _poll():
        global _classify_timer_fn
        import sys as _sys
        pkg = _sys.modules.get(__package__)
        if pkg is None:
            _classify_timer_fn = None
            return None

        rq = pkg.get_result_queue()
        if rq is None:
            _classify_timer_fn = None
            return None

        try:
            result = rq.get_nowait()
        except Exception:
            # Nothing ready yet — check again later.
            return 0.05

        if result.get('type') == 'error':
            utils.log("async", f"worker error: {result.get('msg')}")
            _classify_timer_fn = None
            return None

        if result.get('type') != 'classify_all_result':
            # ping/pong or unknown — put back and stop timer
            _classify_timer_fn = None
            return None

        job_id = result.get('id')
        if job_id != _classify_job_id:
            utils.log("async", f"discarding stale result id={job_id}")
            return 0.05   # keep polling — our result may still be coming

        _apply_classify_result(result)
        _classify_timer_fn = None
        return None  # unregister timer

    _classify_timer_fn = _poll
    bpy.app.timers.register(_poll, first_interval=0.05)


def _cancel_classify_timer():
    global _classify_timer_fn
    if _classify_timer_fn is not None:
        try:
            bpy.app.timers.unregister(_classify_timer_fn)
        except Exception:
            pass
        _classify_timer_fn = None


def _apply_classify_result(result):
    """Apply a classify_all_result from the worker to the local caches.

    Rebuilds hatch/checker batches and triggers a redraw. Called from the
    polling timer — always on the main thread.
    """
    self_results  = result.get('self_results', {})
    cross_results = result.get('cross_results', {})

    # Update self-cache
    for name, entry in self_results.items():
        if name in _obj_cache:
            _isect_self_cache[name] = entry

    # Update cross-cache
    for pair_key, entry in cross_results.items():
        _isect_cross_cache[pair_key] = entry

    # Evict stale entries for objects no longer in cache
    for name in list(_isect_self_cache):
        if name not in _obj_cache:
            del _isect_self_cache[name]
    active_pairs = set()
    names = list(_obj_cache.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            active_pairs.add((na, nb) if na <= nb else (nb, na))
    for pk in list(_isect_cross_cache):
        if pk not in active_pairs:
            del _isect_cross_cache[pk]

    # Check if any active overlay prop is available
    try:
        props = bpy.context.scene.uv_id_props
    except Exception:
        return

    if props.show_intersect and not props.is_muted:
        _rebuild_hatch_from_cache(props)

    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.tag_redraw()
    except Exception:
        pass

    utils.log("async", f"classify result applied, job_id={result.get('id')}")


def _rebuild_hatch_from_cache(props):
    """Rebuild hatch/checker GPU batches from the current classify caches.

    Called after async classify result arrives — does NOT re-run classify.
    Identical logic to the batch-building part of _rebuild_intersect_batches.
    """
    global _inter_island_tris, _inter_gray, _inter_threshold

    shader   = _get_shader()
    opacity  = props.intersect_opacity
    tiled    = (props.intersect_uv_mode == 'TILED')

    all_islands_flat = [
        isle
        for cache in _obj_cache.values()
        if cache.get('islands')
        for isle in cache['islands']
    ]

    global_inter       = set()
    global_stack       = set()
    global_inter_pairs = set()

    obj_names    = list(_obj_cache.keys())
    base_indices = {}
    idx = 0
    for name, cache in _obj_cache.items():
        base_indices[name] = idx
        idx += len(cache.get('islands') or [])

    for name in obj_names:
        entry = _isect_self_cache.get(name)
        if not entry:
            continue
        base = base_indices[name]
        for li in entry.get('inter_idx', ()):
            global_inter.add(base + li)
        for li in entry.get('stack_idx', ()):
            global_stack.add(base + li)
        for la, lb in entry.get('inter_pairs') or ():
            fa, fb = base + la, base + lb
            global_inter_pairs.add((fa, fb) if fa < fb else (fb, fa))

    for i in range(len(obj_names)):
        for j in range(i + 1, len(obj_names)):
            na, nb   = obj_names[i], obj_names[j]
            pair_key = (na, nb) if na <= nb else (nb, na)
            entry    = _isect_cross_cache.get(pair_key)
            if not entry:
                continue
            base_a, base_b = base_indices[na], base_indices[nb]
            for li in entry.get('inter_a', ()):
                global_inter.add(base_a + li)
            for li in entry.get('inter_b', ()):
                global_inter.add(base_b + li)
            for li in entry.get('stack_a', ()):
                global_stack.add(base_a + li)
            for li in entry.get('stack_b', ()):
                global_stack.add(base_b + li)
            for la, lb in entry.get('inter_pairs') or ():
                fa, fb = base_a + la, base_b + lb
                global_inter_pairs.add((fa, fb) if fa < fb else (fb, fa))

    # Tile-crossing detection
    tile_crossing_flat = set()
    if tiled:
        for name, cache in _obj_cache.items():
            islands = cache.get('islands') or []
            base    = base_indices[name]
            for li in ix.find_tile_crossing_islands(islands):
                fi = base + li
                global_inter.add(fi)
                tile_crossing_flat.add(fi)

    # Build hatch/checker batches (reuses seg cache)
    hatch_coords, hatch_colors     = [], []
    checker_coords, checker_colors = [], []
    checker_col = (1.0, 1.0, 1.0, opacity)
    _hits = _miss = 0
    live_keys = {isle.uv_key for isle in all_islands_flat if isle.uv_key is not None}

    for fi, isle in enumerate(all_islands_flat):
        key = isle.uv_key
        if fi in global_inter:
            r, g, b, _ = isle.color
            hc = (r, g, b, opacity)
            if key is not None and key in _hatch_seg_cache:
                segs = _hatch_seg_cache[key]; _hits += 1
            else:
                segs = ix.generate_hatch(isle.tris)
                if key is not None:
                    _hatch_seg_cache[key] = segs
                _miss += 1
            for p1, p2 in segs:
                hatch_coords += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
                hatch_colors  += [hc, hc]
        if fi in global_stack:
            if key is not None and key in _cross_hatch_seg_cache:
                cross_segs = _cross_hatch_seg_cache[key]; _hits += 1
            else:
                cross_segs = ix.generate_cross_hatch(isle.tris)
                if key is not None:
                    _cross_hatch_seg_cache[key] = cross_segs
                _miss += 1
            for p1, p2 in cross_segs:
                checker_coords += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
                checker_colors  += [checker_col, checker_col]

    for dead in [k for k in _hatch_seg_cache       if k not in live_keys]:
        del _hatch_seg_cache[dead]
    for dead in [k for k in _cross_hatch_seg_cache if k not in live_keys]:
        del _cross_hatch_seg_cache[dead]

    def _make(prim, coords, colors):
        return batch_for_shader(shader, prim, {"pos": coords, "color": colors}) if coords else None

    _intersect_batches['hatch']   = _make('LINES', hatch_coords,   hatch_colors)
    _intersect_batches['checker'] = _make('LINES', checker_coords, checker_colors)

    # Offscreen tris
    _build_offscreen_tris(all_islands_flat, global_inter, global_inter_pairs,
                          tile_crossing_flat, tiled)
    utils.log("async", f"hatch rebuilt: hatch_segs={len(hatch_coords)//2} "
              f"stack={len(global_stack)} hits={_hits} miss={_miss}")


def _build_offscreen_tris(all_islands_flat, global_inter, global_inter_pairs,
                          tile_crossing_flat, tiled):
    """Populate _inter_island_tris for the offscreen red fill pass."""
    global _inter_island_tris, _inter_gray, _inter_threshold

    seen_norm_keys = set()
    inter_tris_raw = []
    n_unique       = 0

    def _island_in_tile0(isle):
        cx = (isle.aabb[0] + isle.aabb[2]) * 0.5
        cy = (isle.aabb[1] + isle.aabb[3]) * 0.5
        return (max(0, math.floor(cx)) == 0 and max(0, math.floor(cy)) == 0)

    if tiled:
        for fi in tile_crossing_flat:
            isle = all_islands_flat[fi]
            mn_u, mn_v, mx_u, mx_v = isle.aabb
            touches_tile0 = (mn_u < 1.0 - ix.UV_EPS and mx_u > ix.UV_EPS and
                             mn_v < 1.0 - ix.UV_EPS and mx_v > ix.UV_EPS)
            if touches_tile0:
                inter_tris_raw.append(isle.tris)
                inter_tris_raw.append(isle.tris)
                n_unique += 1

        for fi_a, fi_b in global_inter_pairs:
            if fi_a in tile_crossing_flat or fi_b in tile_crossing_flat:
                continue
            isle_a = all_islands_flat[fi_a]
            isle_b = all_islands_flat[fi_b]
            if not _island_in_tile0(isle_a) and not _island_in_tile0(isle_b):
                continue
            norm_a = ix.normalize_island(isle_a)
            norm_b = ix.normalize_island(isle_b)
            for norm in (norm_a, norm_b):
                key = norm.uv_key
                if key is not None:
                    if key in seen_norm_keys:
                        continue
                    seen_norm_keys.add(key)
                inter_tris_raw.append(norm.tris)
                n_unique += 1
    else:
        for fi, isle in enumerate(all_islands_flat):
            if fi not in global_inter:
                continue
            key = isle.uv_key
            if key is not None:
                if key in seen_norm_keys:
                    continue
                seen_norm_keys.add(key)
            inter_tris_raw.append(isle.tris)
            n_unique += 1

    _inter_gray      = 1.0 / (n_unique + 1) if n_unique > 0 else 0.5
    _inter_threshold = _inter_gray * 1.5
    _inter_island_tris = [tri for tris in inter_tris_raw for tri in tris]
    offscreen.mark_dirty()


def _rebuild_intersect_batches(props):
    """Classify islands and rebuild hatch/checker batches.

    If the background worker is available, dispatches the classify job
    asynchronously and returns immediately — results arrive via _apply_classify_result.
    Falls back to synchronous classify if the worker is not running.
    """
    global _inter_island_tris

    _t0 = time.perf_counter()

    # Try async path first 
    if _dispatch_classify_job(props):
        _start_classify_timer()
        utils.log("async", "classify dispatched to worker — returning immediately")
        return

    utils.log("async", "worker unavailable — falling back to sync classify")

    # Synchronous fallback
    shader   = _get_shader()
    opacity  = props.intersect_opacity
    tiled    = (props.intersect_uv_mode == 'TILED')

    all_islands_flat = [
        isle
        for cache in _obj_cache.values()
        if cache.get('islands')
        for isle in cache['islands']
    ]

    global_inter       = set()
    global_stack       = set()
    global_inter_pairs = set()   # (flat_fi_a, flat_fi_b) for offscreen tile logic

    obj_names    = list(_obj_cache.keys())
    n_objs       = len(obj_names)
    base_indices = {}
    idx = 0
    for name, cache in _obj_cache.items():
        base_indices[name] = idx
        idx += len(cache.get('islands') or [])

    for name in obj_names:
        cache    = _obj_cache[name]
        islands  = cache.get('islands') or []
        cur_hash = cache.get('hash')
        prev     = _isect_self_cache.get(name)
        if prev and prev['uv_hash'] == cur_hash:
            entry = prev
        else:
            p           = prev or {}
            det_islands = [ix.normalize_island(i) for i in islands] if tiled else islands
            inter_idx, stack_idx, uv_kh, i_pairs, island_keys, pair_cache = ix.classify_islands(
                det_islands,
                prev_inter_idx    = p.get('inter_idx'),
                prev_stack_idx    = p.get('stack_idx'),
                prev_uv_key_hash  = p.get('uv_key_hash'),
                prev_inter_pairs  = p.get('inter_pairs'),
                prev_island_keys  = p.get('island_keys'),
                prev_pair_cache   = p.get('pair_cache'),
            )
            entry = {'uv_hash': cur_hash, 'inter_idx': inter_idx,
                     'stack_idx': stack_idx,
                     'uv_key_hash': uv_kh, 'inter_pairs': i_pairs,
                     'island_keys': island_keys, 'pair_cache': pair_cache}
            _isect_self_cache[name] = entry
        base = base_indices[name]
        for li in entry['inter_idx']:
            global_inter.add(base + li)
        for li in entry['stack_idx']:
            global_stack.add(base + li)
        for la, lb in entry.get('inter_pairs') or ():
            fa, fb = base + la, base + lb
            global_inter_pairs.add((fa, fb) if fa < fb else (fb, fa))

    active_pairs = set()
    for i in range(n_objs):
        for j in range(i + 1, n_objs):
            na, nb   = obj_names[i], obj_names[j]
            pair_key = (na, nb) if na <= nb else (nb, na)
            active_pairs.add(pair_key)

            ca, cb = _obj_cache[na], _obj_cache[nb]
            ia, ib = ca.get('islands') or [], cb.get('islands') or []
            ha, hb = ca.get('hash'), cb.get('hash')

            prev = _isect_cross_cache.get(pair_key)
            if prev and prev['ha'] == ha and prev['hb'] == hb:
                entry = prev
            else:
                p      = prev or {}
                det_ia = [ix.normalize_island(i) for i in ia] if tiled else ia
                det_ib = [ix.normalize_island(i) for i in ib] if tiled else ib
                r_a, r_b, s_a, s_b, uv_h, i_pairs, keys_a, keys_b, pair_cache = ix.classify_islands_cross(
                    det_ia, det_ib,
                    prev_inter_a       = p.get('inter_a'),
                    prev_inter_b       = p.get('inter_b'),
                    prev_stack_a       = p.get('stack_a'),
                    prev_stack_b       = p.get('stack_b'),
                    prev_uv_hash       = p.get('uv_hash'),
                    prev_inter_pairs   = p.get('inter_pairs'),
                    prev_island_keys_a = p.get('island_keys_a'),
                    prev_island_keys_b = p.get('island_keys_b'),
                    prev_pair_cache    = p.get('pair_cache'),
                )
                entry = {'ha': ha, 'hb': hb,
                         'inter_a': r_a, 'inter_b': r_b,
                         'stack_a': s_a, 'stack_b': s_b,
                         'uv_hash': uv_h, 'inter_pairs': i_pairs,
                         'island_keys_a': keys_a, 'island_keys_b': keys_b,
                         'pair_cache': pair_cache}
                _isect_cross_cache[pair_key] = entry

            base_a = base_indices[na]
            base_b = base_indices[nb]
            for li in entry['inter_a']:
                global_inter.add(base_a + li)
            for li in entry['inter_b']:
                global_inter.add(base_b + li)
            for li in entry.get('stack_a', ()):
                global_stack.add(base_a + li)
            for li in entry.get('stack_b', ()):
                global_stack.add(base_b + li)
            for la, lb in entry.get('inter_pairs') or ():
                fa, fb = base_a + la, base_b + lb
                global_inter_pairs.add((fa, fb) if fa < fb else (fb, fa))

    for name in list(_isect_self_cache):
        if name not in _obj_cache:
            del _isect_self_cache[name]
    for pk in list(_isect_cross_cache):
        if pk not in active_pairs:
            del _isect_cross_cache[pk]

    # Tile-crossing islands straddle a UDIM boundary — flag as errors, hatch + red fill.
    tile_crossing_flat = set()
    if tiled:
        for name, cache in _obj_cache.items():
            islands = cache.get('islands') or []
            base    = base_indices[name]
            for li in ix.find_tile_crossing_islands(islands):
                fi = base + li
                global_inter.add(fi)
                tile_crossing_flat.add(fi)
        utils.log("rebuild", f"tile_crossing={len(tile_crossing_flat)}")

    hatch_coords, hatch_colors     = [], []
    checker_coords, checker_colors = [], []
    checker_col = (1.0, 1.0, 1.0, opacity)

    _hits = 0
    _miss = 0

    # Collect all uv_keys present this rebuild for cache eviction at the end.
    live_keys = {isle.uv_key for isle in all_islands_flat if isle.uv_key is not None}

    for fi, isle in enumerate(all_islands_flat):
        key = isle.uv_key

        if fi in global_inter:
            r, g, b, _ = isle.color
            hc = (r, g, b, opacity)
            if key is not None and key in _hatch_seg_cache:
                segs = _hatch_seg_cache[key]
                _hits += 1
            else:
                segs = ix.generate_hatch(isle.tris)
                if key is not None:
                    _hatch_seg_cache[key] = segs
                _miss += 1
            for p1, p2 in segs:
                hatch_coords += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
                hatch_colors  += [hc, hc]

        if fi in global_stack:
            if key is not None and key in _cross_hatch_seg_cache:
                cross_segs = _cross_hatch_seg_cache[key]
                _hits += 1
            else:
                cross_segs = ix.generate_cross_hatch(isle.tris)
                if key is not None:
                    _cross_hatch_seg_cache[key] = cross_segs
                _miss += 1
            for p1, p2 in cross_segs:
                checker_coords += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
                checker_colors  += [checker_col, checker_col]

    # Evict entries whose islands no longer exist or have moved (uv_key changed).
    for dead in [k for k in _hatch_seg_cache       if k not in live_keys]:
        del _hatch_seg_cache[dead]
    for dead in [k for k in _cross_hatch_seg_cache if k not in live_keys]:
        del _cross_hatch_seg_cache[dead]

    def _make(prim, coords, colors):
        if not coords:
            return None
        return batch_for_shader(shader, prim, {"pos": coords, "color": colors})

    _intersect_batches['hatch']   = _make('LINES', hatch_coords,   hatch_colors)
    _intersect_batches['checker'] = _make('LINES', checker_coords, checker_colors)

    # Offscreen red-fill tris — tiled mode rules:
    #   Tile-crossing touching tile 0 → raw tris ×2 (guaranteed red fill in place)
    #   Pair with one island in tile 0 → both normalized to tile 0
    #   Pair both in tile 1+ → hatch only, no red fill
    #   UDIM mode → all inter islands, uv_key dedup to avoid double-drawing stacked pairs

    def _island_in_tile0(isle):
        cx = (isle.aabb[0] + isle.aabb[2]) * 0.5
        cy = (isle.aabb[1] + isle.aabb[3]) * 0.5
        return (max(0, math.floor(cx)) == 0 and max(0, math.floor(cy)) == 0)

    seen_norm_keys = set()
    inter_tris_raw = []
    n_unique       = 0   # unique contributors (tile-crossing counts as 1 each)

    if tiled:
        for fi in tile_crossing_flat:
            isle = all_islands_flat[fi]
            mn_u, mn_v, mx_u, mx_v = isle.aabb
            touches_tile0 = (mn_u < 1.0 - ix.UV_EPS and mx_u > ix.UV_EPS and
                             mn_v < 1.0 - ix.UV_EPS and mx_v > ix.UV_EPS)
            if touches_tile0:
                # Twice → 2*gray → guaranteed red fill.
                inter_tris_raw.append(isle.tris)
                inter_tris_raw.append(isle.tris)
                n_unique += 1

        for fi_a, fi_b in global_inter_pairs:
            if fi_a in tile_crossing_flat or fi_b in tile_crossing_flat:
                continue   # tile-crossing already handled above
            isle_a = all_islands_flat[fi_a]
            isle_b = all_islands_flat[fi_b]
            if not _island_in_tile0(isle_a) and not _island_in_tile0(isle_b):
                continue   # both in tile 1+: hatch only
            norm_a = ix.normalize_island(isle_a)
            norm_b = ix.normalize_island(isle_b)
            for norm in (norm_a, norm_b):
                key = norm.uv_key
                if key is not None:
                    if key in seen_norm_keys:
                        continue
                    seen_norm_keys.add(key)
                inter_tris_raw.append(norm.tris)
                n_unique += 1
    else:
        for fi, isle in enumerate(all_islands_flat):
            if fi not in global_inter:
                continue
            key = isle.uv_key
            if key is not None:
                if key in seen_norm_keys:
                    continue
                seen_norm_keys.add(key)
            inter_tris_raw.append(isle.tris)
            n_unique += 1

    global _inter_gray, _inter_threshold
    _inter_gray      = 1.0 / (n_unique + 1) if n_unique > 0 else 0.5
    _inter_threshold = _inter_gray * 1.5

    _inter_island_tris = [tri for tris in inter_tris_raw for tri in tris]
    offscreen.mark_dirty()
    utils.log("rebuild", f"inter_tris={len(_inter_island_tris)}, "
          f"hatch_segs={len(hatch_coords)//2}, stack={len(global_stack)}")
    utils.log("timing", f"intersect_rebuild={1000*(time.perf_counter()-_t0):.1f}ms  "
          f"hatch_segs={len(hatch_coords)//2}  offscreen_tris={len(_inter_island_tris)}  "
          f"cache hits={_hits} miss={_miss}")


def _rebuild_id_opacity(props):
    """Swap alpha in cached ID batch geometry — no reclassification or island re-extraction."""
    alpha  = props.opacity
    shader = _get_shader()
    for cache in _obj_cache.values():
        coords = cache.get('id_coords')
        rgba   = cache.get('id_rgba')
        if not coords or not rgba:
            cache['id_batch'] = None
            continue
        colors = [(r, g, b, alpha) for r, g, b, _ in rgba]
        cache['id_batch'] = batch_for_shader(
            shader, 'TRIS', {"pos": coords, "color": colors}
        )


def _rebuild_intersect_opacity(props):
    """Rebuild hatch/checker batches with new opacity — reuses cached classification and hatch geometry.

    Does NOT re-run classify_islands / classify_islands_cross.
    """
    if not _obj_cache:
        return

    opacity     = props.intersect_opacity
    shader      = _get_shader()
    checker_col = (1.0, 1.0, 1.0, opacity)

    # Reconstruct flat island list and base indices (same logic as _rebuild_intersect_batches).
    all_islands_flat = [
        isle
        for cache in _obj_cache.values()
        if cache.get('islands')
        for isle in cache['islands']
    ]

    obj_names    = list(_obj_cache.keys())
    base_indices = {}
    idx = 0
    for name, cache in _obj_cache.items():
        base_indices[name] = idx
        idx += len(cache.get('islands') or [])

    # Pull global_inter / global_stack straight from the existing classify caches.
    global_inter = set()
    global_stack = set()

    for name in obj_names:
        entry = _isect_self_cache.get(name)
        if not entry:
            continue
        base = base_indices[name]
        for li in entry.get('inter_idx', ()):
            global_inter.add(base + li)
        for li in entry.get('stack_idx', ()):
            global_stack.add(base + li)

    for pair_key, entry in _isect_cross_cache.items():
        na, nb = pair_key
        if na not in base_indices or nb not in base_indices:
            continue
        base_a = base_indices[na]
        base_b = base_indices[nb]
        for li in entry.get('inter_a', ()):
            global_inter.add(base_a + li)
        for li in entry.get('inter_b', ()):
            global_inter.add(base_b + li)
        for li in entry.get('stack_a', ()):
            global_stack.add(base_a + li)
        for li in entry.get('stack_b', ()):
            global_stack.add(base_b + li)

    hatch_coords,   hatch_colors   = [], []
    checker_coords, checker_colors = [], []

    for fi, isle in enumerate(all_islands_flat):
        key = isle.uv_key
        if fi in global_inter:
            r, g, b, _ = isle.color
            hc   = (r, g, b, opacity)
            segs = _hatch_seg_cache.get(key) if key else None
            if segs is None:
                segs = ix.generate_hatch(isle.tris)
                if key:
                    _hatch_seg_cache[key] = segs
            for p1, p2 in segs:
                hatch_coords += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
                hatch_colors  += [hc, hc]
        if fi in global_stack:
            cross = _cross_hatch_seg_cache.get(key) if key else None
            if cross is None:
                cross = ix.generate_cross_hatch(isle.tris)
                if key:
                    _cross_hatch_seg_cache[key] = cross
            for p1, p2 in cross:
                checker_coords += [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
                checker_colors  += [checker_col, checker_col]

    def _make(prim, coords, colors):
        return batch_for_shader(shader, prim, {"pos": coords, "color": colors}) if coords else None

    _intersect_batches['hatch']   = _make('LINES', hatch_coords,   hatch_colors)
    _intersect_batches['checker'] = _make('LINES', checker_coords, checker_colors)

def _rebuild_padding_batches(props):
    padding.rebuild(props, _obj_cache)


def update_batches_safe(context):
    global is_calculating, _obj_cache

    if is_calculating:
        return
    is_calculating = True
    _t0 = time.perf_counter()

    try:
        props        = context.scene.uv_id_props
        uv_id_mode   = props.overlay_mode
        uv_id_alpha  = props.opacity
        any_changed  = False
        active_names = set()

        # Sort so colour assignment divides the hue wheel by global total first.
        edit_objs = sorted(
            [o for o in context.scene.objects
             if o.type == 'MESH' and o.mode == 'EDIT'],
            key=lambda o: o.name
        )
        total_objs = len(edit_objs)

        # Pre-count connected groups across all objects so the global palette
        # gives maximally separated hues across every edit-mode object.
        group_offsets       = {}
        precomp_groups      = {}
        total_global_groups = 0
        if uv_id_mode == 'CONNECTED':
            for obj in edit_objs:
                try:
                    bm_live = bmesh.from_edit_mesh(obj.data)
                    # Read-only — safe on bm_live directly, no copy needed.
                    groups = _mesh_connected_groups(bm_live)
                    precomp_groups[obj.name] = groups
                    n = len(groups)
                except Exception:
                    precomp_groups[obj.name] = _PREPASS_FAILED
                    n = 1
                group_offsets[obj.name] = total_global_groups
                total_global_groups    += n
            total_global_groups = max(total_global_groups, 1)

        for obj_index, obj in enumerate(edit_objs):
            active_names.add(obj.name)

            new_hash, new_id_batch, new_islands, new_id_coords, new_id_rgba = _build_obj_data(
                obj, uv_id_mode, uv_id_alpha, obj_index, total_objs,
                group_offset=group_offsets.get(obj.name, 0),
                total_global_groups=total_global_groups,
                precomputed_groups=precomp_groups.get(obj.name),
            )

            if new_islands is not None:
                _obj_cache[obj.name] = {
                    'hash':      new_hash,
                    'id_batch':  new_id_batch,
                    'islands':   new_islands,
                    'id_coords': new_id_coords,   # raw (x,y,z) list — used by _rebuild_id_opacity
                    'id_rgba':   new_id_rgba,     # raw (r,g,b,a) list — alpha swapped on opacity change
                    'tex_w':     float(obj.uv_id_props.tex_res_x),
                    'tex_h':     float(obj.uv_id_props.tex_res_y),
                    'target_texel': float(obj.uv_id_props.stretch_internal_texel),
                }
                # Do NOT pop _isect_self_cache or _isect_cross_cache here!
                # The worker needs the previous cache state to do per-island pair-cache diffs.
                any_changed = True
            elif new_hash is not None and obj.name not in _obj_cache:
                any_changed = True

        for name in list(_obj_cache):
            if name not in active_names:
                del _obj_cache[name]
                _isect_self_cache.pop(name, None)
                for pk in list(_isect_cross_cache):
                    if name in pk:
                        del _isect_cross_cache[pk]
                any_changed = True

        if props.show_intersect and not props.is_muted:
            if any_changed or (_intersect_batches['hatch'] is None and _classify_timer_fn is None):
                _rebuild_intersect_batches(props)
        else:
            _intersect_batches['hatch'] = None
            _intersect_batches['checker'] = None

        if props.show_padding and not props.is_muted:
            if any_changed or padding.batches['ok'] is None:
                _rebuild_padding_batches(props)
        else:
            padding.clear()

        if props.show_stretch and not props.is_muted:
            if any_changed or stretch._geo_batch is None:
                stretch.rebuild(props, _obj_cache, context)
        else:
            stretch.clear()

    except Exception as e:
        utils.log("update", f"error: {e}")
    finally:
        is_calculating = False
        utils.log("timing", f"update_batches={1000*(time.perf_counter()-_t0):.1f}ms")


@persistent
def depsgraph_update_handler(scene, depsgraph):
    prop = getattr(scene, "uv_id_props", None)
    if not prop or prop.is_muted:
        return
    if not prop.show_uv_id and not prop.show_intersect and not prop.show_padding and not prop.show_stretch:
        return

    if bpy.context.mode != 'EDIT_MESH':
        if _obj_cache:
            _obj_cache.clear()
            _isect_self_cache.clear()
            _isect_cross_cache.clear()
            _intersect_batches['hatch']   = None
            _intersect_batches['checker'] = None
            padding.clear()
            stretch.clear()
        # Clear tris and mark dirty so the red fill disappears on leaving edit mode.
        global _inter_island_tris
        _inter_island_tris = []
        offscreen.mark_dirty()
        _cancel_debounce()
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.tag_redraw()
        except Exception:
            pass
        return

    # If the cache is empty but we are in edit mode and overlays are enabled,
    # it means we just entered edit mode (or just enabled an overlay).
    # We must force a rebuild even if is_updated_geometry is false.
    force_rebuild = False
    if not _obj_cache and (prop.show_uv_id or prop.show_intersect or prop.show_padding or prop.show_stretch):
        force_rebuild = True

    if not force_rebuild and not any(u.is_updated_geometry and isinstance(u.id, bpy.types.Mesh)
                                     for u in depsgraph.updates):
        return

    def _do_rebuild():
        if not is_calculating:
            update_batches_safe(bpy.context)
            try:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            area.tag_redraw()
            except Exception:
                pass

    if prop.live_update:
        _do_rebuild()
    else:
        # Empty cache means we just entered edit mode — rebuild immediately
        # rather than waiting for the debounce delay.
        if not _obj_cache:
            _do_rebuild()
        else:
            _schedule_debounce()


def draw_callback():

    space = bpy.context.space_data
    if space and hasattr(space, 'overlay') and not space.overlay.show_overlays:
        return

    props = bpy.context.scene.uv_id_props
    if props.is_muted:
        return

    shader = _get_shader()

    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        shader.bind()

        # Pass 1: stretch overlay (checker + heatmap)
        if props.show_stretch:
            stretch.draw(props, shader, bpy.context)
            shader.bind() # restore shader in case stretch changed it

        # Pass 2: UV ID color fill
        if props.show_uv_id:
            for cache in _obj_cache.values():
                b = cache.get('id_batch')
                if b:
                    b.draw(shader)

        if props.show_intersect:
            # Pass 3: hatch on intersecting islands 
            gpu.state.line_width_set(2.0)
            if _intersect_batches['hatch']:
                _intersect_batches['hatch'].draw(shader)

            # Pass 4: cross-hatch on stacked islands 
            if _intersect_batches['checker']:
                _intersect_batches['checker'].draw(shader)

            # Pass 5: offscreen overlap fill (always drawn if intersections exist)
            if _inter_island_tris:
                utils.log("pass4", f"tris={len(_inter_island_tris)}")
                if offscreen.check_view_matrix():
                    offscreen.mark_dirty()
                offscreen.render(_inter_island_tris, shader, _inter_gray)
                offscreen.composite(props.intersect_opacity, _inter_threshold)
                shader.bind()  # restore after offscreen composite

        # Pass 6: padding outlines
        if props.show_padding:
            if padding.batches['ok']:
                padding.batches['ok'].draw(shader)
            if padding.batches['bad']:
                padding.batches['bad'].draw(shader)

    except Exception as e:
        utils.log("draw", f"error: {e}")
        traceback.print_exc()
    finally:
        # Restore GPU state — exceptions here would corrupt Blender's own rendering.
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.line_width_set(1.0)


def register():
    if depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)
    bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)


def unregister():
    global draw_handler, _shader
    global _inter_island_tris, _inter_gray, _inter_threshold

    if draw_handler:
        bpy.types.SpaceImageEditor.draw_handler_remove(draw_handler, 'WINDOW')
        draw_handler = None

    if depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)

    _cancel_debounce()
    _cancel_classify_timer()

    offscreen.free()
    padding.clear()
    stretch.clear()

    _obj_cache.clear()
    _isect_self_cache.clear()
    _isect_cross_cache.clear()
    _hatch_seg_cache.clear()
    _cross_hatch_seg_cache.clear()
    _intersect_batches['hatch']   = None
    _intersect_batches['checker'] = None
    _inter_island_tris = []
    _inter_gray        = 0.5
    _inter_threshold   = 0.6
    _shader            = None
    utils.log_clear()


def tag_redraw(context):
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()