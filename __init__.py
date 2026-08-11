import importlib
import os
import struct
import pickle
import subprocess
import sys

# Background worker launched via Blender's CLI command mechanism

_worker_process  = None    # subprocess.Popen
_worker_start_time = None
_next_job_id     = 0
_ipc_buffer      = bytearray()
_worker_synced_objects = {}           # {obj_name: hash} tracks worker's mesh cache state
_ipc_synced      = False
_pending_job     = None

_cli_commands = []   # handles returned by bpy.utils.register_cli_command


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
    global _pending_job
    proc = _worker_process
    if proc is None or proc.poll() is not None:
        # Worker is dead — restart it for the next call, bail for this one.
        if proc is not None:
            print(f"[UVO] Worker died (exit={proc.poll()}) — restarting")
        start_worker()
        return False
        
    if not _ipc_synced:
        _pending_job = job
        print("[UVO] Worker starting — job queued for delivery after handshake")
        return True

    try:
        _write_job(proc, job)
        return True
    except Exception as e:
        print(f"[UVO] Worker send error: {e}")
        return False


def _read_pipe_nonblocking(stream):
    """Read whatever bytes are available right now without blocking."""
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
            fd    = stream.fileno()
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
    """Collect all complete result frames currently available. Never blocks."""
    global _ipc_buffer, _ipc_synced, _pending_job
    proc = _worker_process
    if proc is None or proc.poll() is not None:
        return []

    # Drain stderr so worker log lines appear in the Blender console.
    while True:
        chunk = _read_pipe_nonblocking(proc.stderr)
        if not chunk:
            break
        sys.stdout.write(chunk.decode('utf-8', errors='replace'))

    # Drain stdout into the reassembly buffer.
    while True:
        chunk = _read_pipe_nonblocking(proc.stdout)
        if not chunk:
            break
        _ipc_buffer.extend(chunk)

    # If not synced yet, scan for the handshake
    if not _ipc_synced:
        sync_idx = _ipc_buffer.find(b'UVO_SYNC')
        if sync_idx == -1:
            if len(_ipc_buffer) > 7:
                del _ipc_buffer[:-7]
            return []
        del _ipc_buffer[:sync_idx + 8]
        _ipc_synced = True
        
        global _worker_start_time
        if _worker_start_time is not None:
            import time
            elapsed = time.time() - _worker_start_time
            print(f"[UVO] Worker ready in {elapsed:.2f}s")
            _worker_start_time = None
        
        if _pending_job is not None:
            try:
                _write_job(proc, _pending_job)
                print("[UVO] Queued job delivered after handshake")
            except Exception as e:
                print(f"[UVO] Failed to deliver queued job: {e}")
            _pending_job = None

    # Parse complete length-prefixed frames.
    results = []
    while len(_ipc_buffer) >= 4:
        size = struct.unpack('>I', _ipc_buffer[:4])[0]
        if len(_ipc_buffer) < 4 + size:
            break
        frame = _ipc_buffer[4:4 + size]
        del _ipc_buffer[:4 + size]
        try:
            results.append(pickle.loads(frame))
        except Exception as e:
            print(f"[UVO] Worker IPC unpickle error: {e}")

    return results


def start_worker():
    """Spawn the background Blender worker process."""
    global _worker_process, _ipc_buffer, _ipc_synced, _pending_job

    if _worker_process is not None and _worker_process.poll() is None:
        return   # already alive

    clear_synced_objects()
    _ipc_buffer.clear()
    _ipc_synced = False
    _pending_job = None

    import os
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(addon_dir).replace('\\', '/')
    pkg = __package__
    
    expr = f"import sys; sys.path.insert(0, '{parent_dir}'); from {pkg} import worker; worker.main_loop(sys.argv)"
    
    # Launch Blender in background mode without loading other addons or UI
    cmd = [bpy.app.binary_path, '--background', '--factory-startup', '--python-expr', expr]

    if utils._debug_enabled():
        cmd.append('--uvo-debug')

    kwargs = dict(
        stdin  = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | 0x00000200

    try:
        import time
        global _worker_start_time
        _worker_start_time = time.time()
        _worker_process = subprocess.Popen(cmd, **kwargs)
        print(f"[UVO] Worker process started (pid={_worker_process.pid})")
    except Exception as e:
        print(f"[UVO] Failed to start worker: {e}")
        _worker_process = None


def stop_worker():
    """Signal EOF to the worker and wait for it to exit cleanly."""
    global _worker_process

    proc = _worker_process
    if proc is None:
        return

    try:
        proc.stdin.close()
    except Exception:
        pass

    try:
        proc.wait(timeout=.2)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=0.1)
        except Exception:
            pass

    print("[UVO] Worker process stopped")
    _worker_process = None


def _uvo_worker_command(argv):
    """Called when Blender runs: blender --background --command uvo_worker"""
    from . import worker
    return worker.main_loop(argv)


if "bpy" in locals():
    importlib.reload(utils)
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
    importlib.reload(draw_3d)
else:
    from . import utils
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
    from . import draw_3d

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

    # 3D Seams Aesthetics
    seams_3d_color: bpy.props.FloatVectorProperty(
        name="Seam Color",
        subtype='COLOR_GAMMA',
        size=3,
        default=(0.6235, 0.2510, 1.0),
        min=0.0, max=1.0,
        description="Color for UV seam lines in the 3D Viewport",
    )
    seams_3d_opacity: bpy.props.FloatProperty(
        name="Seam Opacity",
        default=0.85,
        min=0.0, max=1.0,
        subtype='FACTOR',
        description="Opacity of UV seam lines in the 3D Viewport",
    )
    seams_3d_style: bpy.props.EnumProperty(
        name="Seam Style",
        items=[
            ('SOLID', "Solid", "Draw continuous solid lines"),
            ('DASHED', "Dash", "Draw dashed lines"),
        ],
        default='SOLID',
        description="Line style for UV seams",
    )

    def draw(self, context):
        layout = self.layout
        
        layout.label(text="UV Seam")
        
        row = layout.row(align=False)
        row.prop(self, "seams_3d_style", text="")
        row.prop(self, "seams_3d_color", text="")
        
        layout.separator()
        layout.prop(self, "debug")


def register():
    bpy.utils.register_class(UVOAddonPreferences)

    _cli_commands.append(
        bpy.utils.register_cli_command("uvo_worker", _uvo_worker_command)
    )

    # Load icons before UI registers.
    pcoll = bpy.utils.previews.new()
    try:
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        pcoll.load("uv_overlay_on",  os.path.join(icons_dir, "uv_overlay_on.png"),  'IMAGE')
        pcoll.load("uv_overlay_off", os.path.join(icons_dir, "uv_overlay_off.png"), 'IMAGE')
        for i in range(12):
            name = f"clock_frame_{i:02d}"
            pcoll.load(name, os.path.join(icons_dir, f"{name}.png"), 'IMAGE')
        
        if not bpy.app.background:
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
    draw_3d.register()
    ui.register()


def unregister():
    stop_worker()

    for cmd in _cli_commands:
        bpy.utils.unregister_cli_command(cmd)
    _cli_commands.clear()

    ui.unregister()
    draw_3d.unregister()
    draw.unregister()
    ops.unregister()
    props.unregister()

    bpy.utils.unregister_class(UVOAddonPreferences)

    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()