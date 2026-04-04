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



def register():
    bpy.utils.register_class(UV_OT_ToggleOverlay)
    bpy.utils.register_class(UV_OT_RefreshOverlay)


def unregister():
    bpy.utils.unregister_class(UV_OT_RefreshOverlay)
    bpy.utils.unregister_class(UV_OT_ToggleOverlay)