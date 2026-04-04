import bpy
import time


class UV_OT_ToggleOverlay(bpy.types.Operator):
    """Toggle mute on all UV ID overlays"""
    bl_idname = "uv.toggle_id_overlay"
    bl_label  = "UV ID Overlays"

    def execute(self, context):
        props = context.scene.uv_id_props
        props.is_muted = not props.is_muted
        return {'FINISHED'}


class UV_OT_RefreshOverlay(bpy.types.Operator):
    """Force recalculate overlay"""
    bl_idname = "uv.refresh_id_overlay"
    bl_label  = "Refresh UV ID Overlay"

    def execute(self, context):
        from . import draw
        draw.full_refresh(context)
        return {'FINISHED'}




class UV_OT_PingWorker(bpy.types.Operator):
    """Send a ping to the background worker and report round-trip time"""
    bl_idname = "uv.ping_worker"
    bl_label  = "Ping Worker"

    def execute(self, context):
        import sys
        pkg = sys.modules.get(__package__)
        if pkg is None:
            self.report({'ERROR'}, "Addon package not found")
            return {'CANCELLED'}

        proc = pkg.get_worker_process()
        if proc is None or proc.poll() is not None:
            self.report({'ERROR'}, "Worker not running — reload the addon")
            return {'CANCELLED'}

        job_id = pkg.next_job_id()
        t0 = time.perf_counter()

        if not pkg.send_job({'id': job_id, 'type': 'ping'}):
            self.report({'ERROR'}, "Failed to send job to worker")
            return {'CANCELLED'}

        try:
            result = pkg.read_result_blocking(timeout=5.0)
            ms = 1000 * (time.perf_counter() - t0)
            if result and result.get('type') == 'pong':
                self.report({'INFO'}, f"Worker pong in {ms:.1f}ms  (job_id={result.get('id')})")
            else:
                self.report({'WARNING'}, f"Unexpected result: {result}")
        except Exception as e:
            self.report({'ERROR'}, f"Worker timeout: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

def register():
    bpy.utils.register_class(UV_OT_ToggleOverlay)
    bpy.utils.register_class(UV_OT_RefreshOverlay)
    bpy.utils.register_class(UV_OT_PingWorker)


def unregister():
    bpy.utils.unregister_class(UV_OT_PingWorker)
    bpy.utils.unregister_class(UV_OT_RefreshOverlay)
    bpy.utils.unregister_class(UV_OT_ToggleOverlay)