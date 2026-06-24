import importlib
import os
import struct
import pickle
import subprocess
import sys

# Background worker (Popen + length-prefixed pickle over stdin/stdout).
# Avoids multiprocessing module-name-resolution issues in Blender extensions.
# Non-blocking IPC via platform-specific pipe peek — no threading required.

_worker_process  = None    # subprocess.Popen
_next_job_id     = 0
_ipc_buffer      = bytearray()
_worker_synced_objects = {}           # {obj_name: hash} tracks worker's mesh cache state


def get_synced_hash(name):
    return _worker_synced_objects.get(name)

def mark_synced(name, obj_hash):
    _worker_synced_objects[name] = obj_hash

def clear_synced_objects():
    _worker_synced_objects.clear()

def evict_stale_synced_objects(active_names):
    """Evict hashes for objects no longer in the active set, matching the worker's cache clearing."""
    stale = [name for name in _worker_synced_objects if name not in active_names]
    for name in stale:
        del _worker_synced_objects[name]


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
        # Worker is dead — restart it for the next call, bail for this one.
        if proc is not None:
            print(f"[UVO] Worker died (exit={proc.poll()}) — restarting")
        start_worker()
        return False
    try:
        _write_job(proc, job)
        return True
    except Exception as e:
        print(f"[UVO] Worker send error: {e}")
        return False


def _read_pipe_nonblocking(stream):
    """Read available bytes from a pipe without blocking. Returns b'' if nothing ready."""
    if sys.platform == 'win32':
        import msvcrt, ctypes
        from ctypes import wintypes
        try:
            handle = msvcrt.get_osfhandle(stream.fileno())
            avail = wintypes.DWORD(0)
            ok = ctypes.windll.kernel32.PeekNamedPipe(
                handle, None, 0, None, ctypes.byref(avail), None
            )
            if not ok or avail.value == 0:
                return b''
            return os.read(stream.fileno(), avail.value)
        except Exception:
            return b''
    else:
        import fcntl
        try:
            fd = stream.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                return os.read(fd, 65536)
            except BlockingIOError:
                return b''
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except Exception:
            return b''


def get_worker_results():
    """Read all available results from the non-blocking worker pipe."""
    global _ipc_buffer
    proc = _worker_process
    if proc is None or proc.poll() is not None:
        return []

    # Read stderr non-blocking
    while True:
        err_chunk = _read_pipe_nonblocking(proc.stderr)
        if not err_chunk:
            break
        sys.stdout.write(err_chunk.decode('utf-8', errors='replace'))

    # Read stdout non-blocking
    while True:
        chunk = _read_pipe_nonblocking(proc.stdout)
        if not chunk:
            break
        _ipc_buffer.extend(chunk)

    results = []
    # Parse buffer
    while True:
        if len(_ipc_buffer) < 4:
            break
        size = struct.unpack('>I', _ipc_buffer[:4])[0]
        if len(_ipc_buffer) < 4 + size:
            break
        
        data = _ipc_buffer[4:4+size]
        try:
            result = pickle.loads(data)
            results.append(result)
        except Exception as e:
            print(f"[UVO] Worker IPC unpickle error: {e}")
            
        del _ipc_buffer[:4+size]
        
    return results


def start_worker():
    """Spawn the background worker subprocess. Safe to call multiple times."""
    global _worker_process, _ipc_buffer

    if _worker_process is not None and _worker_process.poll() is None:
        return   # already alive

    clear_synced_objects()
    _ipc_buffer.clear()

    addon_dir  = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(addon_dir, "worker.py")

    if not os.path.exists(worker_script):
        print(f"[UVO] worker.py not found at {worker_script}")
        return

    cmd = [sys.executable] + list(bpy.app.python_args) + [worker_script, addon_dir]
    
    if utils._debug_enabled():
        cmd.append("--debug")

    try:
        _worker_process = subprocess.Popen(
            cmd,
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        print(f"[UVO] Worker process started (pid={_worker_process.pid})")
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


if "bpy" in locals():
    importlib.reload(utils)
    importlib.reload(worker)
    importlib.reload(offscreen)
    importlib.reload(intersect)
    importlib.reload(padding)
    importlib.reload(stretch_checker)
    importlib.reload(stretch_heatmap)
    importlib.reload(stretch)
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
    from . import stretch_checker
    from . import stretch_heatmap
    from . import stretch
    from . import props
    from . import ops
    from . import draw
    from . import ui

import bpy
import bpy.utils.previews

preview_collections = {}


def update_debug_pref(self, context):
    """Restart the worker dynamically if the user changes the debug toggle."""
    try:
        from . import utils
        utils._cached_debug = self.debug
    except Exception:
        pass
    stop_worker()
    start_worker()


class UVOAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    debug: bpy.props.BoolProperty(
        name="Debug Logging",
        description=(
            "Print [UVO] debug messages to the system console, "
            "including worker file logs and rebuild timing"
        ),
        default=False,
        update=update_debug_pref
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "debug")


def register():
    bpy.utils.register_class(UVOAddonPreferences)
    # Load icons before UI registers.
    pcoll = bpy.utils.previews.new()
    try:
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        pcoll.load("uv_overlay_on",  os.path.join(icons_dir, "uv_overlay_on.png"),  'IMAGE')
        pcoll.load("uv_overlay_off", os.path.join(icons_dir, "uv_overlay_off.png"), 'IMAGE')
        for i in range(12):
            name = f"clock_frame_{i:02d}"
            pcoll.load(name, os.path.join(icons_dir, f"{name}.png"), 'IMAGE')
        
        # Force decode to prevent lazy-load spinner on first toggle.
        _ = pcoll["uv_overlay_on"].icon_id
        _ = pcoll["uv_overlay_on"].image_size
        _ = pcoll["uv_overlay_off"].icon_id
        _ = pcoll["uv_overlay_off"].image_size
        for i in range(12):
            icon = pcoll[f"clock_frame_{i:02d}"]
            _ = icon.icon_id
            _ = icon.image_size
            
        preview_collections["main"] = pcoll
    except Exception as e:
        print(f"[UVO] Warning: custom icons unavailable ({e}) — using fallback icon")
        bpy.utils.previews.remove(pcoll)

    props.register()
    ops.register()
    draw.register()
    ui.register()


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