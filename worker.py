"""
Background worker process for UVO overlay computation.

Run as standalone script: python worker.py <addon_dir>
Communication: stdin/stdout with length-prefixed pickle frames.

Job types:
    ping     -> pong
    compute  -> compute_result   (unified: classify + stretch in one pass)
"""

import sys
import os
import struct
import pickle
import math
import threading
import time
import traceback

# PROTECT IPC PIPE FROM ROGUE PRINTS:
# Blender does not drain stderr. Never print() directly to it, or the 
# 4KB OS buffer will fill up and permanently freeze the worker thread.
ipc_out = sys.stdout.buffer
sys.stdout = sys.stderr

# Worker-side timeout.
JOB_TIMEOUT_SECS = 10.0

_LOG_PATH = None
_log_lock = threading.Lock()
DEBUG_MODE = False


def _wlog(msg):
    if not DEBUG_MODE or not _LOG_PATH:
        return
        
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with _log_lock:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass




def _read_job(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack('>I', header)[0]
    data = stream.read(size)
    if len(data) < size:
        return None
    return pickle.loads(data)


def _write_result(stream, result):
    data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack('>I', len(data)))
    stream.write(data)
    stream.flush()




def _deserialize_island(d, ix):
    """Reconstruct an intersect.Island from a serialized dict."""
    flat_tris = d['flat_tris']
    tris = [
        ((ft[0], ft[1]), (ft[2], ft[3]), (ft[4], ft[5]))
        for ft in flat_tris
    ]
    flat_segs = d['flat_segs']
    boundary_segs = [
        ((fs[0], fs[1]), (fs[2], fs[3]))
        for fs in flat_segs
    ]
    isle               = ix.Island(tris, d['color'], d['object_name'])
    isle.boundary_segs = boundary_segs
    isle.uv_key        = d['uv_key']
    isle.jacobians     = d.get('jacobians', [])
    isle.uv_area       = d.get('uv_area', 0.0)
    isle.surface_area  = d.get('surface_area', 0.0)
    # aabb is computed by Island.__init__ from tris, but we restore the
    # serialized value to ensure consistency with the main thread.
    if 'aabb' in d:
        isle.aabb = d['aabb']
    return isle




_worker_mesh_cache = {}  # {name: {'hash': int, 'islands': list, 'det_islands': list}}
_stretch_cache     = {}  # {name: [(cache_key, result), ...]}  per-island stretch results


def _run_classify(objects, cross_prev, tiled, job_id, ix):
    """Run self + cross classify for all objects. Returns (self_results, cross_results)."""
    self_results = {}
    for obj in objects:
        p    = obj['prev_self']
        name = obj['name']
        _wlog(f"job {job_id}: SELF '{name}' ({len(obj['det_islands'])} islands)")
        t1 = time.perf_counter()

        inter_idx, stack_idx, uv_kh, i_pairs, ikeys, pcache = ix.classify_islands(
            obj['det_islands'],
            prev_inter_idx   = p.get('inter_idx'),
            prev_stack_idx   = p.get('stack_idx'),
            prev_uv_key_hash = p.get('uv_key_hash'),
            prev_inter_pairs = p.get('inter_pairs'),
            prev_island_keys = p.get('island_keys'),
            prev_pair_cache  = p.get('pair_cache'),
        )
        _wlog(f"job {job_id}: SELF done '{name}' "
              f"{(time.perf_counter()-t1)*1000:.0f}ms — "
              f"inter={len(inter_idx)} stack={len(stack_idx)}")

        self_results[name] = {
            'uv_hash':     obj['hash'],
            'inter_idx':   inter_idx,
            'stack_idx':   stack_idx,
            'uv_key_hash': uv_kh,
            'inter_pairs': i_pairs,
            'island_keys': ikeys,
            'pair_cache':  pcache,
        }

    cross_results = {}
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            oa, ob   = objects[i], objects[j]
            na, nb   = oa['name'], ob['name']
            pair_key = (na, nb) if na <= nb else (nb, na)
            p        = cross_prev.get(pair_key, {})
            t2       = time.perf_counter()

            r_a, r_b, s_a, s_b, uv_h, i_pairs, ckeys_a, ckeys_b, cpcache = \
                ix.classify_islands_cross(
                    oa['det_islands'], ob['det_islands'],
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
            _wlog(f"job {job_id}: CROSS done '{na}'x'{nb}' "
                  f"{(time.perf_counter()-t2)*1000:.0f}ms — "
                  f"inter_a={len(r_a)} inter_b={len(r_b)}")

            cross_results[pair_key] = {
                'ha': oa['hash'], 'hb': ob['hash'],
                'inter_a': r_a,   'inter_b': r_b,
                'stack_a': s_a,   'stack_b': s_b,
                'uv_hash': uv_h,  'inter_pairs': i_pairs,
                'island_keys_a': ckeys_a, 'island_keys_b': ckeys_b,
                'pair_cache':    cpcache,
            }

    return self_results, cross_results


def _run_stretch(objects, job_id):
    """Compute stretch overlay for all objects. Returns stretch_results dict.

    Uses _stretch_cache for per-island result caching (keyed by uv_key + tex settings).
    _worker_mesh_cache must already be populated before calling.
    """
    stretch_results = {}
    total_cached = total_computed = 0

    for obj in objects:
        name     = obj['name']
        tex_w    = obj.get('tex_w') or 1024.0
        tex_h    = obj.get('tex_h') or 1024.0
        target_tx = obj.get('target_texel') or 0.0

        islands = obj['islands']
        if not islands:
            continue

        prev_by_key = {ck: res for ck, res in _stretch_cache.get(name, [])}

        new_isle_cache     = []
        all_coords         = []
        all_warped         = []
        all_checker_colors = []
        all_heatmap_colors = []

        for isle in islands:
            cache_key     = (isle.uv_key, tex_w, tex_h, target_tx)
            cached_result = prev_by_key.get(cache_key)

            if cached_result is not None:
                total_cached += 1
                r = cached_result
            else:
                total_computed += 1
                co, wuv, chk, hm = _stretch_compute_island(isle, tex_w, tex_h, target_tx)
                r = {'coords': co, 'warped_uvs': wuv,
                     'checker_colors': chk, 'heatmap_colors': hm}

            new_isle_cache.append((cache_key, r))
            all_coords.extend(r['coords'])
            all_warped.extend(r['warped_uvs'])
            all_checker_colors.extend(r['checker_colors'])
            all_heatmap_colors.extend(r['heatmap_colors'])

        _stretch_cache[name] = new_isle_cache
        stretch_results[name] = {
            'coords':         all_coords,
            'warped_uvs':     all_warped,
            'checker_colors': all_checker_colors,
            'heatmap_colors': all_heatmap_colors,
        }

    _wlog(f"job {job_id}: stretch computed={total_computed} cached={total_cached}")
    return stretch_results


def _handle_compute(job, ix):
    """Unified handler: sync mesh cache, then run classify and/or stretch.

    Step 1 — Delta-IPC: deserialize any changed island data into _worker_mesh_cache.
    Step 2 — Classify:  run if do_classify=True, returns self/cross results.
    Step 3 — Stretch:   run if do_stretch=True, returns per-object vertex data.

    Both computations share the same (already-synced) island data, so there is
    no race between them and no stale-data risk.
    """
    obj_data    = job.get('objects', [])
    job_id      = job.get('id', '?')
    tiled       = job.get('tiled', True)
    cross_prev  = job.get('cross_prev', {})
    do_classify = job.get('do_classify', False)
    do_stretch  = job.get('do_stretch', False)

    t0 = time.perf_counter()

    # ── Step 1: Sync _worker_mesh_cache (Delta-IPC) ──────────────────────────
    active_names = {od['name'] for od in obj_data}
    for name in list(_worker_mesh_cache):
        if name not in active_names:
            del _worker_mesh_cache[name]
    for name in list(_stretch_cache):
        if name not in active_names:
            del _stretch_cache[name]

    objects = []
    for od in obj_data:
        raw_islands = od.get('islands')
        name        = od['name']
        h           = od['hash']

        if raw_islands is not None:
            islands     = [_deserialize_island(d, ix) for d in raw_islands]
            det_islands = [ix.normalize_island(i) for i in islands] if tiled else islands
            _worker_mesh_cache[name] = {'hash': h, 'islands': islands,
                                        'det_islands': det_islands}
            _wlog(f"job {job_id}: synced '{name}' ({len(islands)} islands)")
        else:
            cached = _worker_mesh_cache.get(name)
            if cached and cached['hash'] == h:
                islands     = cached['islands']
                det_islands = cached['det_islands']
            else:
                _wlog(f"job {job_id}: cache miss '{name}' hash={h}")
                islands     = []
                det_islands = []

        objects.append({
            'name':        name,
            'hash':        h,
            'islands':     islands,
            'det_islands': det_islands,
            'prev_self':   od.get('prev_self', {}),
            'tex_w':       od.get('tex_w'),
            'tex_h':       od.get('tex_h'),
            'target_texel': od.get('target_texel'),
        })

    _wlog(f"job {job_id}: sync done {(time.perf_counter()-t0)*1000:.0f}ms "
          f"— {len(objects)} objs classify={do_classify} stretch={do_stretch}")

    result = {'id': job_id, 'type': 'compute_result'}

    # ── Step 2: Classify ──────────────────────────────────────────────────────
    if do_classify:
        self_r, cross_r = _run_classify(objects, cross_prev, tiled, job_id, ix)
        result['self_results']  = self_r
        result['cross_results'] = cross_r

    # ── Step 3: Stretch ───────────────────────────────────────────────────────
    if do_stretch:
        result['stretch_results'] = _run_stretch(objects, job_id)

    _wlog(f"job {job_id}: COMPLETE {(time.perf_counter()-t0)*1000:.0f}ms total")
    return result


def _process_job(job, ix):
    job_type = job.get('type')

    if job_type == 'ping':
        return {'id': job.get('id'), 'type': 'pong'}

    if job_type == 'compute':
        return _handle_compute(job, ix)

    return {'id': job.get('id'), 'type': 'error', 'msg': f'unknown: {job_type!r}'}


_STRETCH_COL_BLUE = (0.0, 0.0, 1.0, 1.0)
_STRETCH_COL_GRAY = (0.214, 0.214, 0.214, 0.0)
_STRETCH_COL_RED  = (1.0, 0.0, 0.0, 1.0)


def _compute_vertex_jacobians(isle):
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


def _compute_heat_colors(vert_M_sum, vert_area_sum, scale_u, scale_v, mode):
    """Compute per-vertex heat color from Jacobian data.

    mode='checker' uses the additive formula with boosted magnitude.
    mode='heatmap' uses the weighted formula.
    """
    col_blue = _STRETCH_COL_BLUE
    col_gray = _STRETCH_COL_GRAY
    col_red  = _STRETCH_COL_RED
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
            t = -val
            c = (
                col_gray[0] + (col_blue[0] - col_gray[0]) * t,
                col_gray[1] + (col_blue[1] - col_gray[1]) * t,
                col_gray[2] + (col_blue[2] - col_gray[2]) * t,
                col_gray[3] + (col_blue[3] - col_gray[3]) * t,
            )
        else:
            t = val
            c = (
                col_gray[0] + (col_red[0] - col_gray[0]) * t,
                col_gray[1] + (col_red[1] - col_gray[1]) * t,
                col_gray[2] + (col_red[2] - col_gray[2]) * t,
                col_gray[3] + (col_red[3] - col_gray[3]) * t,
            )
        heat_colors[key] = c

    return heat_colors


def _compute_warped_uvs(isle, vert_M_sum, vert_area_sum, scale):
    """BFS integration + Poisson relaxation for checker grid warped UVs."""
    pivot_u = (isle.aabb[0] + isle.aabb[2]) * 0.5
    pivot_v = (isle.aabb[1] + isle.aabb[3]) * 0.5

    # Build adjacency
    adj = {}
    for tri in isle.tris:
        keys = [(round(u, 5), round(v, 5)) for u, v in tri]
        for j in range(3):
            k1, k2 = keys[j], keys[(j+1) % 3]
            if k1 != k2:
                adj.setdefault(k1, set()).add(k2)
                adj.setdefault(k2, set()).add(k1)

    # Find root closest to center
    root_key = None
    min_dist = float('inf')
    for k in vert_M_sum:
        dist = (k[0] - pivot_u)**2 + (k[1] - pivot_v)**2
        if dist < min_dist:
            min_dist = dist
            root_key = k

    # BFS integrate
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

    # Gauss-Seidel relaxation (20 iterations)
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


def _stretch_compute_island(isle, tex_w, tex_h, target_texel):
    """Compute stretch data for a single island. Returns (coords, warped_uvs, heat_colors_checker, heat_colors_heatmap)."""
    if target_texel > 0:
        scale = target_texel / math.sqrt(tex_w * tex_h)
        scale_u = target_texel / tex_w
        scale_v = target_texel / tex_h
    else:
        scale = math.sqrt(isle.uv_area / isle.surface_area) if isle.surface_area > 0 else 1.0
        aspect = tex_h / tex_w if tex_w > 0 else 1.0
        scale_u = scale * math.sqrt(aspect)
        scale_v = scale / math.sqrt(aspect)

    vert_M_sum, vert_area_sum = _compute_vertex_jacobians(isle)
    checker_colors = _compute_heat_colors(vert_M_sum, vert_area_sum, scale_u, scale_v, 'checker')
    heatmap_colors = _compute_heat_colors(vert_M_sum, vert_area_sum, scale_u, scale_v, 'heatmap')
    w_dict = _compute_warped_uvs(isle, vert_M_sum, vert_area_sum, scale)

    coords = []
    warped_uvs = []
    checker_list = []
    heatmap_list = []
    gray = _STRETCH_COL_GRAY

    for tri in isle.tris:
        for u, v in tri:
            key = (round(u, 5), round(v, 5))
            coords.append((u, v, 0.0))
            warped_uvs.append(w_dict.get(key, (u, v)))
            checker_list.append(checker_colors.get(key, gray))
            heatmap_list.append(heatmap_colors.get(key, gray))

    return coords, warped_uvs, checker_list, heatmap_list





def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: worker.py <addon_dir>")

    addon_dir = sys.argv[1]
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)

    # Only setup logging if requested
    global DEBUG_MODE, _LOG_PATH
    if "--debug" in sys.argv:
        DEBUG_MODE = True
        pid = os.getpid()
        _LOG_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), f"uvo_worker_{pid}.log")
        try:
            with open(_LOG_PATH, "w", encoding="utf-8") as f:
                f.write(f"=== UVO worker started pid={pid} ===\n")
            with open(os.path.join(os.environ.get("TEMP", "/tmp"),
                                   "uvo_worker_latest.log.pid"), "w") as f:
                f.write(_LOG_PATH)
        except Exception:
            pass

    _wlog(f"addon_dir={addon_dir}")

    _ix_err = None
    try:
        import intersect as ix
        _wlog("intersect imported OK")
    except ImportError as e:
        ix      = None
        _ix_err = str(e)
        _wlog(f"intersect import FAILED: {e}")
    except Exception as e:
        ix      = None
        _ix_err = f"Unexpected error: {e}"
        _wlog(f"intersect import FAILED (unexpected): {e}")

    stdin = sys.stdin.buffer

    while True:
        job = _read_job(stdin)
        if job is None:
            _wlog("stdin EOF — exiting")
            break

        job_id   = job.get('id', '?')
        job_type = job.get('type', '?')
        _wlog(f"received job id={job_id} type={job_type!r}")

        # Hard timeout — hung classify exits the process so __init__.py can restart
        result_box = [None]
        error_box  = [None]

        def _run():
            try:
                if ix is None:
                    result_box[0] = {'id': job_id, 'type': 'error',
                                     'msg': f'intersect import failed: {_ix_err}'}
                else:
                    result_box[0] = _process_job(job, ix)
            except Exception as e:
                error_box[0] = (str(e), traceback.format_exc())

        t       = threading.Thread(target=_run, daemon=True)
        t_start = time.perf_counter()
        t.start()
        t.join(timeout=JOB_TIMEOUT_SECS)

        if t.is_alive():
            elapsed = time.perf_counter() - t_start
            msg = (f"TIMEOUT: job id={job_id} type={job_type!r} "
                   f"hung {elapsed:.1f}s — worker exiting for restart")
            _wlog(f"*** {msg} ***")
            try:
                _write_result(ipc_out, {'id': job_id, 'type': 'error', 'msg': msg})
            except Exception:
                pass
            sys.exit(1)   # triggers restart in __init__.send_job

        if error_box[0]:
            err_msg, tb = error_box[0]
            _wlog(f"job {job_id} ERROR: {err_msg}")
            _write_result(ipc_out, {
                'id': job_id, 'type': 'error', 'msg': err_msg, 'tb': tb
            })
        else:
            _write_result(ipc_out, result_box[0])


if __name__ == '__main__':
    main()