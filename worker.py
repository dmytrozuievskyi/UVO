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

# ── PROTECT IPC PIPE FROM ROGUE PRINTS ────────────────────────────────────────
ipc_out = sys.stdout.buffer
sys.stdout = sys.stderr
# ──────────────────────────────────────────────────────────────────────────────


# ── Pipe I/O ──────────────────────────────────────────────────────────────────

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


# ── Island reconstruction ─────────────────────────────────────────────────────

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
    isle.uv_key        = d['uv_key']      # frozenset — picklable
    # aabb and tri_centers already computed in Island.__init__ from tris
    return isle


def _serialize_island(isle):
    """Pack an Island into plain types for pipe transfer."""
    flat_tris = [
        (t[0][0], t[0][1], t[1][0], t[1][1], t[2][0], t[2][1])
        for t in isle.tris
    ]
    flat_segs = [
        (s[0][0], s[0][1], s[1][0], s[1][1])
        for s in isle.boundary_segs
    ]
    return {
        'flat_tris':   flat_tris,
        'flat_segs':   flat_segs,
        'color':       isle.color,
        'object_name': isle.object_name,
        'uv_key':      isle.uv_key,
    }


# ── classify_all handler ──────────────────────────────────────────────────────

_worker_mesh_cache = {}  # {name: {'hash': int, 'islands': list, 'det_islands': list}}


def _handle_classify_all(job, ix):
    """Run full self + cross classify for all objects in the job.

    Job fields:
        tiled   : bool
        objects : list of {name, hash, islands:[serialized], prev_self:{...}}
        cross_prev : {(na,nb): {...}}  — previous cross-cache entries

    Returns classify_all_result with self_results and cross_results.
    """
    tiled    = job.get('tiled', True)
    obj_data = job.get('objects', [])
    cross_prev = job.get('cross_prev', {})

    # Deserialize all islands
    objects = []
    active_names = {od['name'] for od in obj_data}
    for name in list(_worker_mesh_cache.keys()):
        if name not in active_names:
            del _worker_mesh_cache[name]

    for od in obj_data:
        raw_islands = od.get('islands')
        name = od['name']
        h = od['hash']

        if raw_islands is not None:
            islands = [_deserialize_island(d, ix) for d in raw_islands]
            if tiled:
                det_islands = [ix.normalize_island(i) for i in islands]
            else:
                det_islands = islands
            _worker_mesh_cache[name] = {'hash': h, 'islands': islands, 'det_islands': det_islands}
        else:
            cached = _worker_mesh_cache.get(name)
            if not cached or cached['hash'] != h:
                print(f"[UVO] Worker error: cache miss for {name} hash={h}", file=sys.stderr)
                islands = []
                det_islands = []
            else:
                islands = cached['islands']
                det_islands = cached['det_islands']

        objects.append({
            'name':        name,
            'hash':        h,
            'islands':     islands,
            'det_islands': det_islands,
            'prev_self':   od.get('prev_self', {}),
        })

    # Self-classify each object
    self_results = {}
    for obj in objects:
        p  = obj['prev_self']
        inter_idx, stack_idx, uv_kh, i_pairs, ikeys, pcache = ix.classify_islands(
            obj['det_islands'],
            prev_inter_idx   = p.get('inter_idx'),
            prev_stack_idx   = p.get('stack_idx'),
            prev_uv_key_hash = p.get('uv_key_hash'),
            prev_inter_pairs = p.get('inter_pairs'),
            prev_island_keys = p.get('island_keys'),
            prev_pair_cache  = p.get('pair_cache'),
        )
        self_results[obj['name']] = {
            'uv_hash':    obj['hash'],
            'inter_idx':  inter_idx,
            'stack_idx':  stack_idx,
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
            cross_results[pair_key] = {
                'ha': oa['hash'], 'hb': ob['hash'],
                'inter_a': r_a,   'inter_b': r_b,
                'stack_a': s_a,   'stack_b': s_b,
                'uv_hash': uv_h,  'inter_pairs': i_pairs,
                'island_keys_a': ckeys_a, 'island_keys_b': ckeys_b,
                'pair_cache': cpcache,
            }

    return {
        'id':           job.get('id'),
        'type':         'classify_all_result',
        'self_results': self_results,
        'cross_results': cross_results,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def _process_job(job, ix):
    job_id   = job.get('id')
    job_type = job.get('type')

    if job_type == 'ping':
        return {'id': job_id, 'type': 'pong'}

    if job_type == 'classify_all':
        return _handle_classify_all(job, ix)

    return {'id': job_id, 'type': 'error', 'msg': f'unknown: {job_type!r}'}


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: worker.py <addon_dir>")

    addon_dir = sys.argv[1]
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)

    # Import intersect once — stays loaded for the lifetime of the process.
    try:
        import intersect as ix
    except ImportError as e:
        # Fallback: try as a package submodule name
        ix = None
        _ix_err = str(e)

    stdin  = sys.stdin.buffer
    stdout = ipc_out

    while True:
        job = _read_job(stdin)
        if job is None:
            break

        try:
            if ix is None:
                result = {'id': job.get('id'), 'type': 'error',
                          'msg': f'intersect import failed: {_ix_err}'}
            else:
                result = _process_job(job, ix)
        except Exception as e:
            import traceback
            result = {'id': job.get('id'), 'type': 'error',
                      'msg': str(e), 'tb': traceback.format_exc()}

        _write_result(stdout, result)


if __name__ == '__main__':
    main()