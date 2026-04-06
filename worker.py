"""
Background worker process for UV island classification.

Run as standalone script: python worker.py <addon_dir>
Communication: stdin/stdout with length-prefixed pickle frames.

Job types:
    ping          -> pong
    classify_all  -> classify_all_result
"""

import sys
import os
import struct
import pickle
import threading
import time

# PROTECT IPC PIPE FROM ROGUE PRINTS:
# Blender does not drain stderr. Never print() directly to it, or the 
# 4KB OS buffer will fill up and permanently freeze the worker thread.
ipc_out = sys.stdout.buffer
sys.stdout = sys.stderr

# Worker-side timeout. Slightly less than the draw.py watchdog (8 s) so the
# worker can write an informative error before being killed externally.
JOB_TIMEOUT_SECS = 6.5

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


# Pipe I/O 

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


# Island reconstruction 

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
    return isle


# classify_all handler 

_worker_mesh_cache = {}  # {name: {'hash': int, 'islands': list, 'det_islands': list}}


def _handle_classify_all(job, ix):
    """Run full self + cross classify for all objects in the job."""
    tiled      = job.get('tiled', True)
    obj_data   = job.get('objects', [])
    cross_prev = job.get('cross_prev', {})
    job_id     = job.get('id', '?')

    t0 = time.perf_counter()

    # Deserialize / pull from cache 
    active_names = {od['name'] for od in obj_data}
    for name in list(_worker_mesh_cache.keys()):
        if name not in active_names:
            del _worker_mesh_cache[name]

    objects = []
    for od in obj_data:
        raw_islands = od.get('islands')
        name = od['name']
        h    = od['hash']

        if raw_islands is not None:
            islands = [_deserialize_island(d, ix) for d in raw_islands]
            det_islands = [ix.normalize_island(i) for i in islands] if tiled else islands
            _worker_mesh_cache[name] = {'hash': h, 'islands': islands,
                                        'det_islands': det_islands}
        else:
            cached = _worker_mesh_cache.get(name)
            if not cached or cached['hash'] != h:
                _wlog(f"job {job_id}: cache miss for '{name}' hash={h}")
                islands     = []
                det_islands = []
            else:
                islands     = cached['islands']
                det_islands = cached['det_islands']

        objects.append({
            'name':        name,
            'hash':        h,
            'islands':     islands,
            'det_islands': det_islands,
            'prev_self':   od.get('prev_self', {}),
        })

    _wlog(f"job {job_id}: deserialize done {(time.perf_counter()-t0)*1000:.0f}ms "
          f"— {len(objects)} objs tiled={tiled}")

    # Self-classify each object 
    self_results = {}
    for obj in objects:
        p    = obj['prev_self']
        name = obj['name']
        n_isl = len(obj['det_islands'])
        _wlog(f"job {job_id}: SELF '{name}' ({n_isl} islands, "
              f"has_prev={p.get('inter_idx') is not None})")
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

    # Cross-classify each pair
    cross_results = {}
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            oa, ob = objects[i], objects[j]
            na, nb = oa['name'], ob['name']
            pair_key = (na, nb) if na <= nb else (nb, na)
            p = cross_prev.get(pair_key, {})
            _wlog(f"job {job_id}: CROSS '{na}'({len(oa['det_islands'])}) x "
                  f"'{nb}'({len(ob['det_islands'])}) "
                  f"has_prev={p.get('inter_a') is not None}")
            t2 = time.perf_counter()

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

    _wlog(f"job {job_id}: COMPLETE {(time.perf_counter()-t0)*1000:.0f}ms total")

    return {
        'id':            job_id,
        'type':          'classify_all_result',
        'self_results':  self_results,
        'cross_results': cross_results,
    }


# Main loop

def _process_job(job, ix):
    job_type = job.get('type')

    if job_type == 'ping':
        return {'id': job.get('id'), 'type': 'pong'}

    if job_type == 'classify_all':
        return _handle_classify_all(job, ix)

    return {'id': job.get('id'), 'type': 'error', 'msg': f'unknown: {job_type!r}'}


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

    try:
        import intersect as ix
        _wlog("intersect imported OK")
    except ImportError as e:
        ix      = None
        _ix_err = str(e)
        _wlog(f"intersect import FAILED: {e}")

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
                import traceback
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