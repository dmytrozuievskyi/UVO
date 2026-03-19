import bpy


def _sync_draw(context):
    from . import draw

    space = context.space_data
    props = context.scene.uv_id_props

    native_on  = (
        space is not None
        and hasattr(space, 'overlay')
        and space.overlay.show_overlays
    )
    addon_on   = not props.is_muted
    any_active = props.show_uv_id or props.show_intersect or props.show_padding

    should_draw = native_on and addon_on and any_active

    if should_draw:
        if not draw.draw_handler:
            draw.draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                draw.draw_callback, (), 'WINDOW', 'POST_VIEW'
            )
        draw.update_batches_safe(context)
    else:
        if draw.draw_handler:
            bpy.types.SpaceImageEditor.draw_handler_remove(draw.draw_handler, 'WINDOW')
            draw.draw_handler = None
        # Mute only removes the handler — cache stays so unmute redraws immediately.
        if not props.is_muted:
            draw._obj_cache.clear()
            draw._intersect_batches['hatch']   = None
            draw._intersect_batches['checker'] = None

    if context.screen:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def update_mute(self, context):
    from . import draw
    if not self.is_muted:
        draw.full_refresh(context)
    _sync_draw(context)


def update_uv_id(self, context):
    # Mode changed — full rebuild needed to reassign colours.
    from . import draw
    draw.full_refresh(context)
    _sync_draw(context)


def update_uv_id_opacity(self, context):
    """Opacity only — swap alpha in cached batches, no reclassification."""
    from . import draw
    if not self.is_muted and self.show_uv_id:
        draw._rebuild_id_opacity(self)
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def update_intersect_opacity(self, context):
    """Intersect opacity only — recolour cached hatch geometry, no reclassification."""
    from . import draw
    if not self.is_muted and self.show_intersect:
        draw._rebuild_intersect_opacity(self)
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def update_show_uv_id(self, context):
    _sync_draw(context)


def update_intersect(self, context):
    from . import draw
    if self.show_intersect and not self.is_muted:
        draw._rebuild_intersect_batches(self)
    _sync_draw(context)


def update_intersect_settings(self, context):
    # Mode or opacity changed — full rebuild needed.
    from . import draw
    draw.full_refresh(context)


def update_padding(self, context):
    # Rebuild immediately from cached islands — don't wait for geometry change.
    from . import draw
    if self.show_padding and not self.is_muted:
        draw._rebuild_padding_batches(self)
    _sync_draw(context)


def update_padding_settings(self, context):
    from . import draw
    if self.show_padding and not self.is_muted:
        draw._rebuild_padding_batches(self)
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def update_padding_res_x(self, context):
    if self.padding_res_linked:
        self.padding_res_y = self.padding_res_x
    update_padding_settings(self, context)


def update_padding_res_y(self, context):
    update_padding_settings(self, context)


class UVIDProperties(bpy.types.PropertyGroup):

    is_muted: bpy.props.BoolProperty(
        default=False,
        name="Mute Overlays",
        description="Temporarily hide all overlays without changing settings",
        update=update_mute,
    )
    live_update: bpy.props.BoolProperty(
        default=False,
        name="Live Update",
        description=(
            "Update overlays on every UV change.\n"
            "Disable for smoother editing on complex meshes — "
            "overlays will update once after each action completes"
        ),
    )
    show_uv_id: bpy.props.BoolProperty(
        default=False,
        name="UV ID",
        description="Color each UV island distinctly",
        update=update_show_uv_id,
    )
    opacity: bpy.props.FloatProperty(
        default=0.5,
        min=0.0,
        max=1.0,
        name="Opacity",
        subtype='FACTOR',
        description="UV ID overlay transparency",
        update=update_uv_id_opacity,
    )
    overlay_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('OBJECT',    "Object", "One color per object"),
            ('CONNECTED', "Island", "One color per topologically connected island"),
        ],
        description="How to assign colors to UV islands",
        update=update_uv_id,
    )
    show_intersect: bpy.props.BoolProperty(
        default=False,
        name="Intersect",
        description="Highlight overlapping UV islands",
        update=update_intersect,
    )
    intersect_opacity: bpy.props.FloatProperty(
        default=0.85,
        min=0.0,
        max=1.0,
        name="Opacity",
        subtype='FACTOR',
        description="Intersect overlay opacity",
        update=update_intersect_opacity,
    )
    intersect_uv_mode: bpy.props.EnumProperty(
        name="UV Mode",
        items=[
            ('TILED', "Tiled",
             "Fold all UV tiles into (0,1) before detection — "
             "finds overlaps between islands placed in different tiles "
             "that share the same texel space. Use for tiling/repeating textures."),
            ('UDIM',  "UDIM",
             "Each UDIM tile is independent — no coordinate folding. "
             "Use when each tile bakes to a separate texture."),
        ],
        default='TILED',
        update=update_intersect_settings,
    )
    show_padding: bpy.props.BoolProperty(
        default=False,
        name="Padding",
        description="Show padding zones — highlights islands whose padding "
                    "areas overlap, indicating potential mipmap bleed",
        update=update_padding,
    )
    TEXTURE_RES_ITEMS = [
        ('256',  "256",  ""),
        ('512',  "512",  ""),
        ('1024', "1024", ""),
        ('2048', "2048", ""),
        ('4096', "4096", ""),
        ('8192', "8192", ""),
    ]
    padding_res_x: bpy.props.EnumProperty(
        name="Width",
        items=TEXTURE_RES_ITEMS,
        default='2048',
        description="Texture width used to calculate padding size in UV space",
        update=update_padding_res_x,
    )
    padding_res_y: bpy.props.EnumProperty(
        name="Height",
        items=TEXTURE_RES_ITEMS,
        default='2048',
        description="Texture height used to calculate padding size in UV space",
        update=update_padding_res_y,
    )
    padding_res_linked: bpy.props.BoolProperty(
        default=True,
        name="Link Resolutions",
        description="Keep X and Y texture resolution in sync",
        update=update_padding_settings,
    )
    padding_px: bpy.props.EnumProperty(
        name="Padding",
        items=[
            ('2',  "2 px",  ""),
            ('4',  "4 px",  ""),
            ('6',  "6 px",  ""),
            ('8',  "8 px",  ""),
            ('10', "10 px", ""),
            ('12', "12 px", ""),
            ('16', "16 px", ""),
            ('24', "24 px", ""),
            ('32', "32 px", ""),
        ],
        default='4',
        description="Minimum padding distance between islands in pixels",
        update=update_padding_settings,
    )


def register():
    try:
        bpy.utils.unregister_class(UVIDProperties)
    except RuntimeError:
        pass
    bpy.utils.register_class(UVIDProperties)
    bpy.types.Scene.uv_id_props = bpy.props.PointerProperty(type=UVIDProperties)


def unregister():
    try:
        del bpy.types.Scene.uv_id_props
    except AttributeError:
        pass
    try:
        bpy.utils.unregister_class(UVIDProperties)
    except RuntimeError:
        pass