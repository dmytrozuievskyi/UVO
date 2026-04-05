import importlib
import os
import queue as _queue_mod
import struct
import pickle
import subprocess
import sys
import threading

# ── Background worker process ─────────────────────────────────────────────────
# Uses subprocess.Popen + stdin/stdout pipes with length-prefixed pickle frames.
# This avoids multiprocessing module-name-resolution issues in Blender extensions.

_worker_process  = None    # subprocess.Popen
_worker_lock     = threading.Lock()
_next_job_id     = 0
_result_queue    = _queue_mod.Queue()  # worker results arrive here
_reader_thread   = None                # background thread reading worker stdout
_classify_generation = 0              # incremented each time a new classify job is sent
_worker_synced_objects = {}           # {obj_name: hash} tracks worker's mesh cache state


def get_synced_hash(name):
    return _worker_synced_objects.get(name)

def mark_synced(name, obj_hash):
    _worker_synced_objects[name] = obj_hash

def clear_synced_objects():
    _worker_synced_objects.clear()


def _write_job(proc, job):
    """Send a job dict to the worker via its stdin pipe."""
    data = pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.write(struct.pack('>I', len(data)))
    proc.stdin.write(data)
    proc.stdin.flush()


def get_worker_process():
    return _worker_process


def next_job_id():
    global _next_job_id
    _next_job_id += 1
    return _next_job_id


def send_job(job):
    """Send a job to the worker. Returns False if worker not running."""
    proc = _worker_process
    if proc is None or proc.poll() is not None:
        return False
    try:
        with _worker_lock:
            _write_job(proc, job)
        return True
    except Exception as e:
        print(f"[UVO] Worker send error: {e}")
        return False


def read_result_blocking(timeout=5.0):
    """Read one result from the result queue (blocking, for ping test)."""
    try:
        return _result_queue.get(timeout=timeout)
    except _queue_mod.Empty:
        raise TimeoutError("Worker result timeout")



def get_result_queue():
    """Return the queue where worker results arrive."""
    return _result_queue


def get_classify_generation():
    return _classify_generation


def next_classify_generation():
    global _classify_generation
    _classify_generation += 1
    return _classify_generation


def _start_reader_thread():
    """Spawn a daemon thread that reads worker stdout and puts results in _result_queue."""
    global _reader_thread

    proc = _worker_process
    if proc is None:
        return

    def _reader():
        while True:
            try:
                header = proc.stdout.read(4)
                if len(header) < 4:
                    break
                size = struct.unpack('>I', header)[0]
                data = proc.stdout.read(size)
                if len(data) < size:
                    break
                result = pickle.loads(data)
                _result_queue.put(result)
            except Exception:
                break

    _reader_thread = threading.Thread(target=_reader, daemon=True, name='UVO_Reader')
    _reader_thread.start()


def _start_stderr_reader(proc):
    """Spawn a daemon thread to read worker stderr and print it to the Blender console."""
    def _reader_stderr():
        for line in iter(proc.stderr.readline, b''):
            try:
                msg = line.decode('utf-8').rstrip()
                if msg:
                    print(msg)
            except Exception:
                pass

    t = threading.Thread(target=_reader_stderr, daemon=True, name='UVO_Worker_Stderr')
    t.start()


def start_worker():
    """Spawn the background worker subprocess. Safe to call multiple times."""
    global _worker_process

    if _worker_process is not None and _worker_process.poll() is None:
        return   # already alive

    clear_synced_objects()

    addon_dir  = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(addon_dir, "worker.py")

    if not os.path.exists(worker_script):
        print(f"[UVO] worker.py not found at {worker_script}")
        return

    python_exe = sys.executable

    try:
        _worker_process = subprocess.Popen(
            [python_exe, worker_script, addon_dir],
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        print(f"[UVO] Worker process started (pid={_worker_process.pid})")
        _start_reader_thread()
        _start_stderr_reader(_worker_process)
    except Exception as e:
        print(f"[UVO] Failed to start worker: {e}")
        _worker_process = None


def stop_worker():
    """Close the worker's stdin (signals EOF) and wait for it to exit."""
    global _worker_process

    proc = _worker_process
    if proc is None:
        return

    try:
        proc.stdin.close()
    except Exception:
        pass

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=1.0)

    print("[UVO] Worker process stopped")
    _worker_process = None


if "bpy" in locals():  # Support F8 / Reload Scripts
    importlib.reload(utils)
    importlib.reload(worker)
    importlib.reload(offscreen)
    importlib.reload(intersect)
    importlib.reload(padding)
    importlib.reload(props)
    importlib.reload(ops)
    importlib.reload(draw)
    importlib.reload(ui)
else:
    from . import utils
    from . import worker
    from . import offscreen
    from . import intersect
    from . import padding
    from . import props
    from . import ops
    from . import draw
    from . import ui

import bpy
import bpy.utils.previews

preview_collections = {}


class UVOAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    debug: bpy.props.BoolProperty(
        name="Debug Logging",
        description=(
            "Print [UVO] debug messages to the system console, "
            "including rebuild timing"
        ),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "debug")


def register():
    bpy.utils.register_class(UVOAddonPreferences)
    # Icons must load before UI registers — falls back to 'GROUP_UVS' if missing.
    pcoll = bpy.utils.previews.new()
    try:
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        pcoll.load("uv_overlay_on",  os.path.join(icons_dir, "uv_overlay_on.png"),  'IMAGE')
        pcoll.load("uv_overlay_off", os.path.join(icons_dir, "uv_overlay_off.png"), 'IMAGE')
        # Force immediate decode — prevents the lazy-load spinner on first toggle.
        _ = pcoll["uv_overlay_on"].icon_id
        _ = pcoll["uv_overlay_off"].icon_id
        preview_collections["main"] = pcoll
    except Exception as e:
        print(f"[UVO] Warning: custom icons unavailable ({e}) — using fallback icon")
        bpy.utils.previews.remove(pcoll)

    props.register()
    ops.register()
    draw.register()
    ui.register()
    start_worker()


def unregister():
    stop_worker()
    ui.unregister()
    draw.unregister()
    ops.unregister()
    props.unregister()

    bpy.utils.unregister_class(UVOAddonPreferences)

    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()