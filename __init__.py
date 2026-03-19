if "bpy" in locals():  # Support F8 / Reload Scripts
    import importlib
    importlib.reload(utils)
    importlib.reload(offscreen)
    importlib.reload(intersect)
    importlib.reload(padding)
    importlib.reload(props)
    importlib.reload(ops)
    importlib.reload(draw)
    importlib.reload(ui)
else:
    from . import utils
    from . import offscreen
    from . import intersect
    from . import padding
    from . import props
    from . import ops
    from . import draw
    from . import ui

import bpy
import bpy.utils.previews
import os

preview_collections = {}


class UVOAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    complex_intersection: bpy.props.BoolProperty(
        name="Complex Intersection Highlight",
        description=(
            "Show a red fill over overlapping UV areas using an offscreen GPU buffer. "
        ),
        default=False,
    )

    debug: bpy.props.BoolProperty(
        name="Debug Logging",
        description=(
            "Print [UVO] debug messages to the system console. "
        ),
        default=False,
    )

    def draw(self, context):
        self.layout.prop(self, "complex_intersection")
        self.layout.prop(self, "debug")


def register():
    try:
        bpy.utils.unregister_class(UVOAddonPreferences)
    except RuntimeError:
        pass
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


def unregister():
    ui.unregister()
    draw.unregister()
    ops.unregister()
    props.unregister()

    bpy.utils.unregister_class(UVOAddonPreferences)

    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()