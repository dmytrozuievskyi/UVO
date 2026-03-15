import bpy


class IMAGE_PT_uv_id_overlay(bpy.types.Panel):
    bl_label       = "UV ID Overlay"
    bl_idname      = "IMAGE_PT_uv_id_overlay"
    bl_space_type  = 'IMAGE_EDITOR'
    bl_region_type = 'HEADER'
    bl_ui_units_x  = 14

    def draw(self, context):
        layout = self.layout
        props  = context.scene.uv_id_props
        space  = context.space_data

        native_on = hasattr(space, 'overlay') and space.overlay.show_overlays
        if not native_on:
            layout.label(text="Enable Overlays first", icon='INFO')
            return

        if context.mode != 'EDIT_MESH':
            layout.label(text="Enter Edit Mode to use", icon='INFO')
            return

        active = not props.is_muted

        layout.label(text="UV ID")

        row  = layout.row(align=False)
        row.prop(props, "show_uv_id", text="")

        sub  = row.row(align=False)
        sub.enabled = props.show_uv_id and active
        sub.prop(props, "overlay_mode", text="")

        sub2 = row.row(align=False)
        sub2.enabled = props.show_uv_id and active
        sub2.prop(props, "opacity", text="", slider=True)

        layout.label(text="Intersect")

        row3 = layout.row(align=False)
        row3.prop(props, "show_intersect", text="")

        sub3 = row3.row(align=False)
        sub3.enabled = props.show_intersect and active
        sub3.prop(props, "intersect_uv_mode", text="")

        sub4 = row3.row(align=False)
        sub4.enabled = props.show_intersect and active
        sub4.prop(props, "intersect_opacity", text="", slider=True)

        layout.label(text="Padding")

        pad_row = layout.row(align=True)
        pad_row.prop(props, "show_padding", text="")

        sub = pad_row.row(align=True)
        sub.enabled = props.show_padding and active

        sub.separator()
        sub.prop(props, "padding_res_x", text="")

        link_icon = 'LINKED' if props.padding_res_linked else 'UNLINKED'
        sub.prop(props, "padding_res_linked", text="", icon=link_icon, toggle=True)

        res_y_row = sub.row(align=True)
        res_y_row.enabled = not props.padding_res_linked
        res_y_row.prop(props, "padding_res_y", text="")

        sub.separator()
        sub.label(text="Padding:")
        sub.prop(props, "padding_px", text="")

        layout.separator()
        layout.prop(props, "live_update", text="Live Update")


def draw_header_button(self, context):
    if context.space_data.type != 'IMAGE_EDITOR':
        return

    layout    = self.layout
    props     = context.scene.uv_id_props
    space     = context.space_data
    native_on = hasattr(space, 'overlay') and space.overlay.show_overlays
    is_active = (not props.is_muted) and native_on

    import sys
    pkg   = sys.modules.get(__package__)
    pcoll = pkg.preview_collections.get("main") if pkg else None

    layout.separator_spacer()

    row = layout.row(align=True)
    btn = row.row(align=True)
    btn.active = native_on

    if pcoll:
        icon_id = (pcoll["uv_overlay_on"].icon_id if is_active
                   else pcoll["uv_overlay_off"].icon_id)
        btn.operator(
            "uv.toggle_id_overlay",
            text       = "",
            icon_value = icon_id,
            depress    = is_active,
        )
    else:
        btn.operator(
            "uv.toggle_id_overlay",
            text    = "",
            icon    = 'GROUP_UVS',
            depress = is_active,
        )

    row.popover(panel="IMAGE_PT_uv_id_overlay", text="")


def register():
    bpy.utils.register_class(IMAGE_PT_uv_id_overlay)
    bpy.types.IMAGE_HT_tool_header.append(draw_header_button)


def unregister():
    bpy.types.IMAGE_HT_tool_header.remove(draw_header_button)
    bpy.utils.unregister_class(IMAGE_PT_uv_id_overlay)