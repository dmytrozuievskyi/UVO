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

        # ------------------------------------------------------------------
        # Global: Texture Setup
        # ------------------------------------------------------------------
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

            # Texel Density
            td_row = layout.row(align=True)
            td_row.operator("uv.sample_stretch_texel", text="", icon='EYEDROPPER')
            td_row.separator()
            td_row.prop(obj_props, "stretch_target_texel", text="")
            td_row.prop(obj_props, "stretch_texel_unit", text="")
        else:
            layout.label(text="No active object", icon='INFO')

        layout.separator()

        # ------------------------------------------------------------------
        # UV ID
        # ------------------------------------------------------------------
        layout.label(text="UV ID")

        row = layout.row(align=False)
        row.prop(props, "show_uv_id", text="")

        sub = row.row(align=False)
        sub.enabled = props.show_uv_id and active
        sub.prop(props, "overlay_mode", text="")

        sub2 = row.row(align=False)
        sub2.enabled = props.show_uv_id and active
        sub2.prop(props, "opacity", text="", slider=True)

        # ------------------------------------------------------------------
        # Intersect
        # ------------------------------------------------------------------
        layout.label(text="Intersect")

        row3 = layout.row(align=False)
        row3.prop(props, "show_intersect", text="")

        sub3 = row3.row(align=False)
        sub3.enabled = props.show_intersect and active
        sub3.prop(props, "intersect_uv_mode", text="")

        sub4 = row3.row(align=False)
        sub4.enabled = props.show_intersect and active
        sub4.prop(props, "intersect_opacity", text="", slider=True)

        # ------------------------------------------------------------------
        # Padding (Combined into Intersect)
        # ------------------------------------------------------------------
        pad_row = layout.row(align=False)
        pad_row.prop(props, "show_padding", text="")

        pad_sub = pad_row.row(align=False)
        pad_sub.enabled = props.show_padding and active
        pad_sub.separator()
        pad_sub.label(text="Padding:")
        pad_sub.prop(props, "padding_px", text="")

        # ------------------------------------------------------------------
        # Stretch
        # ------------------------------------------------------------------
        layout.label(text="Stretch")

        # Row 2 — toggle | mode | opacity
        stretch_row = layout.row(align=False)
        stretch_row.prop(props, "show_stretch", text="")

        stretch_sub = stretch_row.row(align=False)
        stretch_sub.enabled = props.show_stretch and active
        stretch_sub.prop(props, "stretch_mode", text="")

        stretch_sub2 = stretch_row.row(align=False)
        stretch_sub2.enabled = props.show_stretch and active
        stretch_sub2.prop(props, "stretch_opacity", text="", slider=True)

        # ------------------------------------------------------------------
        # Footer
        # ------------------------------------------------------------------
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