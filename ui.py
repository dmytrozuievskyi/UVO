import bpy


class IMAGE_PT_uv_id_overlay(bpy.types.Panel):
    bl_label       = "UV ID Overlay"
    bl_idname      = "IMAGE_PT_uv_id_overlay"
    bl_space_type  = 'IMAGE_EDITOR'
    bl_region_type = 'HEADER'
    bl_ui_units_x  = 12

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

        layout.label(text="Texture Setup")

        obj_props = context.active_object.uv_id_props if context.active_object else None

        if obj_props:
            tex_row = layout.row(align=True)
            tex_row.prop(obj_props, "tex_res_x", text="")

            link_icon = 'LINKED' if obj_props.tex_res_linked else 'UNLINKED'
            tex_row.prop(obj_props, "tex_res_linked", text="", icon=link_icon, toggle=True)

            res_y_sub = tex_row.row(align=True)
            res_y_sub.enabled = not obj_props.tex_res_linked
            res_y_sub.prop(obj_props, "tex_res_y", text="")

            td_row = layout.row(align=True)
            td_row.operator("uv.sample_stretch_texel", text="", icon='EYEDROPPER')
            td_row.separator()
            td_row.prop(obj_props, "stretch_target_texel", text="")
            td_row.prop(obj_props, "stretch_texel_unit", text="")
        else:
            layout.label(text="No active object", icon='INFO')

        layout.separator()

        layout.label(text="UV ID")

        row_uv = layout.row(align=False)
        row_uv.prop(props, "show_uv_id", text="")

        content_uv = row_uv.row(align=False)
        content_uv.enabled = props.show_uv_id and active


        split_fac = 0.5

        split_uv = content_uv.split(factor=split_fac, align=False)
        split_uv.prop(props, "overlay_mode", text="")
        split_uv.prop(props, "opacity", text="", slider=True)

        layout.label(text="Intersect")

        row_int = layout.row(align=False)

        row_int.prop(props, "show_intersect", text="")

        content_int = row_int.row(align=False)
        content_int.enabled = props.show_intersect and active

        split_int = content_int.split(factor=split_fac, align=False)

        split_int.prop(props, "intersect_uv_mode", text="")
        split_int.prop(props, "intersect_opacity", text="", slider=True)

        row_pad = layout.row(align=False)

        row_pad.prop(props, "show_padding", text="")

        content_pad = row_pad.row(align=False)
        content_pad.enabled = props.show_padding and active

        split_pad = content_pad.split(factor=split_fac, align=False)

        lbl_row = split_pad.row()
        lbl_row.alignment = 'LEFT'
        lbl_row.label(text=" Padding:")

        split_pad.prop(props, "padding_px", text="")

        layout.label(text="Stretch")

        row_str = layout.row(align=False)
        row_str.prop(props, "show_stretch", text="")

        content_str = row_str.row(align=False)
        content_str.enabled = props.show_stretch and active

        split_str = content_str.split(factor=split_fac, align=False)
        split_str.prop(props, "stretch_mode", text="")
        split_str.prop(props, "stretch_opacity", text="", slider=True)

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