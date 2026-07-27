"""
Background worker process for UVO overlay computation.

Run as background Blender process: blender --background --command uvo_worker
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
import time
import traceback

from . import intersect as ix
from . import stretch

# Redirect stdout to stderr so print() doesn't corrupt the IPC pipe.
ipc_out = None

def _init_ipc():
    global ipc_out
    ipc_out = sys.stdout.buffer
    sys.stdout = sys.stderr

DEBUG_MODE = False


def _wlog(msg):
    if not DEBUG_MODE:
        return
        
    ts   = time.strftime("%H:%M:%S")
    # Worker's sys.stdout is redirected to sys.stderr, which is captured
    # by the main Blender process and printed to the system console.
    print(f"[{ts}] {msg}")




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
    isle.local_key     = d.get('local_key')
    isle.ref_a         = d.get('ref_a', (0.0, 0.0))
    isle.ref_b         = d.get('ref_b', (0.0, 0.0))
    isle.jacobians     = d.get('jacobians', [])
    isle.uv_area       = d.get('uv_area', 0.0)
    isle.surface_area  = d.get('surface_area', 0.0)
    if 'aabb' in d:
        isle.aabb = d['aabb']
    return isle




_worker_mesh_cache = {}  # {name: {'hash': int, 'islands': list, 'det_islands': list}}
_stretch_cache     = {}  # {name: [(cache_key, result), ...]}  per-island stretch results
_stretch_local_cache = {}  # {name: {local_cache_key: {'ref_a': A, 'ref_b': B, 'result': ...}}}


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


def _compute_rigid_transform(old_a, old_b, new_a, new_b):
    old_dx = old_b[0] - old_a[0]
    old_dy = old_b[1] - old_a[1]
    new_dx = new_b[0] - new_a[0]
    new_dy = new_b[1] - new_a[1]
    
    old_len = math.sqrt(old_dx**2 + old_dy**2)
    new_len = math.sqrt(new_dx**2 + new_dy**2)
    scale_ratio = new_len / old_len if old_len > 1e-10 else 1.0
    
    denom = old_len * new_len
    if denom < 1e-10:
        return scale_ratio, (1.0, 0.0, 0.0, 1.0, new_a[0] - old_a[0], new_a[1] - old_a[1])
    
    cos_t = (old_dx * new_dx + old_dy * new_dy) / denom
    sin_t = (old_dx * new_dy - old_dy * new_dx) / denom
    
    s = scale_ratio
    m00 = s * cos_t
    m01 = -s * sin_t
    m10 = s * sin_t
    m11 = s * cos_t
    tx = new_a[0] - (m00 * old_a[0] + m01 * old_a[1])
    ty = new_a[1] - (m10 * old_a[0] + m11 * old_a[1])
    
    return scale_ratio, (m00, m01, m10, m11, tx, ty)


def _apply_rigid_transform(result, transform):
    m00, m01, m10, m11, tx, ty = transform
    
    new_coords = []
    for u, v, z in result['coords']:
        new_coords.append((m00*u + m01*v + tx, m10*u + m11*v + ty, z))
    
    new_warped = []
    for wu, wv in result['warped_uvs']:
        new_warped.append((m00*wu + m01*wv + tx, m10*wu + m11*wv + ty))
    
    return {
        'coords':         new_coords,
        'warped_uvs':     new_warped,
        'checker_colors': result['checker_colors'],
        'heatmap_colors': result['heatmap_colors'],
    }


def _run_stretch(objects, job_id):
    """Compute stretch overlay for all objects. Returns stretch_results dict.

    Uses _stretch_cache for per-island result caching (keyed by uv_key + tex settings).
    _worker_mesh_cache must already be populated before calling.
    """
    stretch_results = {}
    total_cached = total_transform_hit = total_computed = 0

    for obj in objects:
        name     = obj['name']
        tex_w    = obj.get('tex_w') or 1024.0
        tex_h    = obj.get('tex_h') or 1024.0
        target_tx = obj.get('target_texel') or 0.0

        islands = obj['islands']
        if not islands:
            continue

        prev_by_key = {ck: res for ck, res in _stretch_cache.get(name, [])}
        local_prev_by_key = _stretch_local_cache.get(name, {})

        new_isle_cache       = []
        new_local_isle_cache = {}
        all_coords           = []
        all_warped           = []
        all_checker_colors   = []
        all_heatmap_colors   = []

        for isle in islands:
            cache_key       = (isle.uv_key, tex_w, tex_h, target_tx)
            local_cache_key = (isle.local_key, tex_w, tex_h, target_tx)
            
            cached_result = prev_by_key.get(cache_key)

            if cached_result is not None:
                total_cached += 1
                r = cached_result
                # Keep local cache populated for future transform detection
                old_local = local_prev_by_key.get(local_cache_key)
                if old_local:
                    new_local_isle_cache[local_cache_key] = old_local
                else:
                    new_local_isle_cache[local_cache_key] = {
                        'ref_a': isle.ref_a, 'ref_b': isle.ref_b,
                        'result': r
                    }
            elif local_cache_key in local_prev_by_key:
                old_entry = local_prev_by_key[local_cache_key]
                scale_ratio, transform = _compute_rigid_transform(
                    old_entry['ref_a'], old_entry['ref_b'], isle.ref_a, isle.ref_b)
                
                # Verify transform using cached coords to avoid false positives
                is_valid = True
                m00, m01, m10, m11, tx, ty = transform
                old_coords = old_entry['result']['coords']
                if len(old_coords) != len(isle.tris) * 3:
                    is_valid = False
                else:
                    idx = 0
                    for tri in isle.tris:
                        for nu, nv in tri:
                            ou, ov, _ = old_coords[idx]
                            tu = m00*ou + m01*ov + tx
                            tv = m10*ou + m11*ov + ty
                            if abs(tu - nu) > 1e-4 or abs(tv - nv) > 1e-4:
                                is_valid = False
                                break
                            idx += 1
                        if not is_valid: break
                
                if is_valid and abs(scale_ratio - 1.0) < 0.001:
                    total_transform_hit += 1
                    r = _apply_rigid_transform(old_entry['result'], transform)
                    
                    new_local_isle_cache[local_cache_key] = {
                        'ref_a': isle.ref_a, 'ref_b': isle.ref_b,
                        'result': r
                    }
                else:
                    total_computed += 1
                    co, wuv, chk, hm = _stretch_compute_island(isle, tex_w, tex_h, target_tx)
                    r = {'coords': co, 'warped_uvs': wuv,
                         'checker_colors': chk, 'heatmap_colors': hm}
                    
                    new_local_isle_cache[local_cache_key] = {
                        'ref_a': isle.ref_a, 'ref_b': isle.ref_b,
                        'result': r
                    }
            else:
                total_computed += 1
                co, wuv, chk, hm = _stretch_compute_island(isle, tex_w, tex_h, target_tx)
                r = {'coords': co, 'warped_uvs': wuv,
                     'checker_colors': chk, 'heatmap_colors': hm}
                
                new_local_isle_cache[local_cache_key] = {
                    'ref_a': isle.ref_a, 'ref_b': isle.ref_b,
                    'result': r
                }

            new_isle_cache.append((cache_key, r))
            all_coords.extend(r['coords'])
            all_warped.extend(r['warped_uvs'])
            all_checker_colors.extend(r['checker_colors'])
            all_heatmap_colors.extend(r['heatmap_colors'])

        _stretch_cache[name] = new_isle_cache
        _stretch_local_cache[name] = new_local_isle_cache
        stretch_results[name] = {
            'coords':         all_coords,
            'warped_uvs':     all_warped,
            'checker_colors': all_checker_colors,
            'heatmap_colors': all_heatmap_colors,
        }

    _wlog(f"job {job_id}: stretch computed={total_computed} cached={total_cached} transform={total_transform_hit}")
    return stretch_results


def _handle_compute(job, ix):
    """Sync mesh cache (Delta-IPC), then run classify and/or stretch."""
    obj_data    = job.get('objects', [])
    job_id      = job.get('id', '?')
    tiled       = job.get('tiled', True)
    cross_prev  = job.get('cross_prev', {})
    do_classify = job.get('do_classify', False)
    do_stretch  = job.get('do_stretch', False)

    t0 = time.perf_counter()

    # Sync _worker_mesh_cache (Delta-IPC)
    active_names = {od['name'] for od in obj_data}
    for name in list(_worker_mesh_cache):
        if name not in active_names:
            del _worker_mesh_cache[name]
    for name in list(_stretch_cache):
        if name not in active_names:
            del _stretch_cache[name]
    for name in list(_stretch_local_cache):
        if name not in active_names:
            del _stretch_local_cache[name]

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

    # Classify
    if do_classify:
        self_r, cross_r = _run_classify(objects, cross_prev, tiled, job_id, ix)
        result['self_results']  = self_r
        result['cross_results'] = cross_r

    # Stretch
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





def _stretch_compute_island(isle, tex_w, tex_h, target_texel):
    """Compute stretch data for a single island. Returns (coords, warped_uvs, heat_colors_checker, heat_colors_heatmap)."""
    scale, scale_u, scale_v = stretch.compute_scale_factors(isle, tex_w, tex_h, target_texel)

    vert_M_sum, vert_area_sum = stretch.compute_vertex_jacobians(isle)

    checker_colors = {}
    heatmap_colors = {}
    for key, area in vert_area_sum.items():
        if area > 1e-8:
            M_avg = [m / area for m in vert_M_sum[key]]
        else:
            M_avg = [1.0, 0.0, 0.0, 1.0]
        
        area_err, angle_err = stretch.compute_stretch_metrics(M_avg, scale_u, scale_v)
        checker_colors[key] = stretch.error_to_color(area_err, angle_err, 'checker')
        heatmap_colors[key] = stretch.error_to_color(area_err, angle_err, 'heatmap')

    w_dict = stretch.compute_warped_uvs(isle, vert_M_sum, vert_area_sum, scale)

    coords = []
    warped_uvs = []
    checker_list = []
    heatmap_list = []
    gray = stretch.COL_GRAY

    for tri in isle.tris:
        for u, v in tri:
            key = (round(u, 5), round(v, 5))
            coords.append((u, v, 0.0))
            warped_uvs.append(w_dict.get(key, (u, v)))
            checker_list.append(checker_colors.get(key, gray))
            heatmap_list.append(heatmap_colors.get(key, gray))

    return coords, warped_uvs, checker_list, heatmap_list





def main_loop(argv):
    """Entry point called by the registered CLI command."""
    _init_ipc()
    
    # Send handshake so parent knows the pipe is now free
    if ipc_out:
        try:
            ipc_out.write(b'UVO_SYNC')
            ipc_out.flush()
        except Exception:
            pass

    global DEBUG_MODE
    if "--uvo-debug" in argv:
        DEBUG_MODE = True
        pid = os.getpid()
        print(f"=== UVO worker started pid={pid} ===")

    _wlog("worker starting — importing addon modules")
    _wlog("entering job loop")
    stdin = sys.stdin.buffer

    while True:
        job = _read_job(stdin)
        if job is None:
            _wlog("stdin EOF — exiting")
            break

        job_id   = job.get('id', '?')
        job_type = job.get('type', '?')
        _wlog(f"received job id={job_id} type={job_type!r}")

        # Process synchronously without threading timeout.
        # The main Blender process can kill the worker if it hangs.
        result = None
        error = None

        try:
            result = _process_job(job, ix)
        except Exception as e:
            err_msg = str(e)
            tb = traceback.format_exc()
            error = {'id': job_id, 'type': 'error', 'msg': err_msg, 'tb': tb}

        if error:
            _wlog(f"job {job_id} ERROR: {error['msg']}")
            try:
                _write_result(ipc_out, error)
            except Exception:
                pass
        elif result:
            try:
                _write_result(ipc_out, result)
            except Exception:
                pass

    os._exit(0)