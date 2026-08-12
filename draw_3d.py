import bpy
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader

# Module-level state
_viewport_states = {}   # {space_ptr: ViewportState}
_draw_handler_3d = None  # single SpaceView3D draw handler (always registered)

class ViewportState:
    """Per-viewport overlay state."""
    __slots__ = ('enabled', 'native_seams_saved', 'native_seams_original')
    
    def __init__(self):
        self.enabled = False
        self.native_seams_saved = False     # whether we've saved the native state
        self.native_seams_original = True   # the user's original show_edge_seams value

def toggle_viewport(context):
    """Toggle the 3D seams overlay for the current viewport only."""
    space = context.space_data
    if space is None or space.type != 'VIEW_3D':
        return
    
    ptr = space.as_pointer()
    state = _viewport_states.setdefault(ptr, ViewportState())
    state.enabled = not state.enabled
    
    if state.enabled:
        _apply_native_seam_mode(space, state, context)
    else:
        _restore_native_seams(space, state)
    
    # Ensure 3D boundary data exists in cache
    _ensure_3d_boundary_data(context)
    context.area.tag_redraw()

def is_active_in_space(space):
    """Check if the overlay is enabled in the given SpaceView3D."""
    if space is None or space.type != 'VIEW_3D':
        return False
    state = _viewport_states.get(space.as_pointer())
    return state is not None and state.enabled

def _apply_native_seam_mode(space, state, context):
    """Apply OVERLAY or OVERRIDE mode to native seams for this viewport."""
    props = context.scene.uv_3d_seam_props
    
    is_drawing = not props.is_muted and getattr(space.overlay, 'show_overlays', False)
    
    if props.seams_3d_mode == 'OVERRIDE' and is_drawing:
        if not state.native_seams_saved:
            state.native_seams_original = space.overlay.show_edge_seams
            state.native_seams_saved = True
        space.overlay.show_edge_seams = False
    else:  # OVERLAY or not drawing
        _restore_native_seams(space, state)

def _restore_native_seams(space, state):
    """Restore the user's original native seam visibility."""
    if state.native_seams_saved:
        space.overlay.show_edge_seams = state.native_seams_original
        state.native_seams_saved = False

def sync_native_seams_all(context):
    """Re-apply native seam mode across all active viewports (called on mode change)."""
    for ptr, state in _viewport_states.items():
        if not state.enabled:
            continue
        # Find the space by pointer
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and space.as_pointer() == ptr:
                        _apply_native_seam_mode(space, state, context)

def _cleanup_stale_viewports():
    """Remove viewport states for spaces that no longer exist."""
    live_ptrs = set()
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            live_ptrs.add(space.as_pointer())
    except Exception:
        return
    
    stale = [ptr for ptr in _viewport_states if ptr not in live_ptrs]
    for ptr in stale:
        del _viewport_states[ptr]

def _extract_3d_boundary_edges(obj, bm, uv_layer):
    """Extract world-space coordinates of edges that lie on UV island boundaries.
    
    An edge is a UV boundary if:
    - It has < 2 linked faces (mesh boundary -> always a UV boundary), OR
    - The UV coordinates on the two sides don't match (UV split)
    
    Returns: (unselected_coords, selected_coords, active_coords)
    """
    mw = obj.matrix_world
    unsel_coords = []
    sel_coords = []
    act_coords = []
    
    active_edge = None
    if bm.select_history:
        active_elem = bm.select_history.active
        if isinstance(active_elem, bmesh.types.BMEdge):
            active_edge = active_elem
            
    for edge in bm.edges:
        if edge.hide:
            continue
            
        if len(edge.link_faces) < 2:
            # Mesh boundary edge -> always a UV boundary
            co1 = mw @ edge.verts[0].co
            co2 = mw @ edge.verts[1].co
            if edge == active_edge:
                coords = act_coords
            elif edge.select:
                coords = sel_coords
            else:
                coords = unsel_coords
            coords.append(((co1.x, co1.y, co1.z), (co2.x, co2.y, co2.z)))
            continue
        
        if len(edge.link_faces) != 2:
            continue  # non-manifold with 3+ faces, skip
        
        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        
        # Find the loops on this edge for each face
        l1 = l2 = None
        for loop in f1.loops:
            if loop.edge == edge:
                l1 = loop
                break
        for loop in f2.loops:
            if loop.edge == edge:
                l2 = loop
                break
        
        if l1 is None or l2 is None:
            continue
        
        # Compare UV coordinates on both sides of the edge
        uv1a = l1[uv_layer].uv
        uv1b = l1.link_loop_next[uv_layer].uv
        uv2a = l2[uv_layer].uv
        uv2b = l2.link_loop_next[uv_layer].uv
        
        UV_EPS = 1e-4
        match = (
            (abs(uv1a.x - uv2a.x) < UV_EPS and abs(uv1a.y - uv2a.y) < UV_EPS and
             abs(uv1b.x - uv2b.x) < UV_EPS and abs(uv1b.y - uv2b.y) < UV_EPS) or
            (abs(uv1a.x - uv2b.x) < UV_EPS and abs(uv1a.y - uv2b.y) < UV_EPS and
             abs(uv1b.x - uv2a.x) < UV_EPS and abs(uv1b.y - uv2a.y) < UV_EPS)
        )
        
        if not match:
            co1 = mw @ edge.verts[0].co
            co2 = mw @ edge.verts[1].co
            if edge == active_edge:
                coords = act_coords
            elif edge.select:
                coords = sel_coords
            else:
                coords = unsel_coords
            coords.append(((co1.x, co1.y, co1.z), (co2.x, co2.y, co2.z)))
    
    return unsel_coords, sel_coords, act_coords

def _ensure_3d_boundary_data(context):
    """Extract 3D boundary edges for all cached objects, if not already done."""
    from . import draw as _draw
    
    for obj in context.scene.objects:
        if obj.type != 'MESH' or obj.mode != 'EDIT':
            continue
        cache = _draw._obj_cache.get(obj.name)
        if cache is None or cache.get('seam_3d_coords') is not None:
            continue
        
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.verify()
        cache['seam_3d_coords_unsel'], cache['seam_3d_coords_sel'], cache['seam_3d_coords_act'] = _extract_3d_boundary_edges(obj, bm, uv_layer)
        cache['seam_3d_batch_unsel'] = None
        cache['seam_3d_batch_sel'] = None
        cache['seam_3d_batch_act'] = None

def draw_callback_3d():
    """SpaceView3D POST_VIEW draw callback. Runs once per viewport per frame."""
    context = bpy.context
    space = context.space_data
    
    # Per-viewport activation check
    if not is_active_in_space(space):
        return
    
    # Must be in Edit Mode
    if context.mode != 'EDIT_MESH':
        return
        
    if not hasattr(space, 'overlay') or not space.overlay.show_overlays:
        return
        
    if context.scene.uv_3d_seam_props.is_muted:
        return
    
    # Periodic stale viewport cleanup (cheap, small dict)
    _cleanup_stale_viewports()
    
    # Ensure boundary data is extracted
    _ensure_3d_boundary_data(context)
    
    # Get preferences
    prefs = context.preferences.addons[__package__].preferences
    color = (*prefs.seams_3d_color, prefs.seams_3d_opacity)
    
    theme = context.preferences.themes[0]
    edge_width = getattr(theme.view_3d, 'edge_width', 1)
    thickness = (edge_width * 2) + 1
        
    style = prefs.seams_3d_style
    
    # Get the appropriate shader
    shader = _get_3d_shader(style)
    if shader is None:
        return
    
    from . import draw as _draw
    
    try:
        # Depth Testing & X-Ray
        if space.shading.show_xray:
            gpu.state.depth_test_set('NONE')
        else:
            gpu.state.depth_test_set('LESS_EQUAL')
        
        gpu.state.blend_set('ALPHA')
        shader.bind()
        shader.uniform_float("ModelViewMatrix", gpu.matrix.get_model_view_matrix())
        shader.uniform_float("ProjectionMatrix", gpu.matrix.get_projection_matrix())
        
        color_unsel = color
        
        # For selected, shift hue by 60 degrees
        from mathutils import Color
        c_sel = Color((color_unsel[0], color_unsel[1], color_unsel[2]))
        h, s, v = c_sel.hsv
        c_sel.hsv = ((h + (60.0 / 360.0)) % 1.0, s, v)
        color_sel = (c_sel.r, c_sel.g, c_sel.b, color_unsel[3])
        
        # For active, keep the desaturated almost-white look
        color_act = (
            color_unsel[0] * 0.1 + 0.9,
            color_unsel[1] * 0.1 + 0.9,
            color_unsel[2] * 0.1 + 0.9,
            color_unsel[3]
        )
        
        theme = context.preferences.themes[0]
        edge_width = getattr(theme.view_3d, 'edge_width', 1)
        
        def draw_batch(cache_key, batch_key, draw_color, depth_mult, line_width):
            gpu.state.line_width_set(float(line_width))
            shader.uniform_float("color", draw_color)
            shader.uniform_float("depth_bias_multiplier", depth_mult)
            
            for name, cache in _draw._obj_cache.items():
                coords = cache.get(cache_key)
                if not coords:
                    continue
                
                batch = cache.get(batch_key)
                if batch is None:
                    flat_coords = []
                    for co1, co2 in coords:
                        flat_coords.append(co1)
                        flat_coords.append(co2)
                    if flat_coords:
                        batch = batch_for_shader(shader, 'LINES', {"pos": flat_coords})
                    else:
                        batch = False # Use False to cache empty state
                    cache[batch_key] = batch
                
                if batch:
                    batch.draw(shader)
                    
        if style == 'OUTLINE':
            thickness_outer = float(edge_width + 5)
            thickness_inner = float(edge_width)
            
            # Draw thick background
            draw_batch('seam_3d_coords_unsel', 'seam_3d_batch_unsel', color_unsel, 1.0, thickness_outer)
            draw_batch('seam_3d_coords_sel', 'seam_3d_batch_sel', color_unsel, 1.0, thickness_outer)
            draw_batch('seam_3d_coords_act', 'seam_3d_batch_act', color_unsel, 1.0, thickness_outer)
            
            # Draw thin inner line
            wire_color = (*theme.view_3d.wire_edit[:3], color_unsel[3])
            sel_color = (*theme.view_3d.edge_mode_select[:3], color_sel[3])
            act_color = (*theme.view_3d.editmesh_active[:3], color_act[3])
            
            draw_batch('seam_3d_coords_unsel', 'seam_3d_batch_unsel', wire_color, 1.02, thickness_inner)
            draw_batch('seam_3d_coords_sel', 'seam_3d_batch_sel', sel_color, 1.02, thickness_inner)
            draw_batch('seam_3d_coords_act', 'seam_3d_batch_act', act_color, 1.02, thickness_inner)
            
        else:
            thickness = float((edge_width * 2) + 1)
            draw_batch('seam_3d_coords_unsel', 'seam_3d_batch_unsel', color_unsel, 1.0, thickness)
            draw_batch('seam_3d_coords_sel', 'seam_3d_batch_sel', color_sel, 1.0, thickness)
            draw_batch('seam_3d_coords_act', 'seam_3d_batch_act', color_act, 1.0, thickness)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.line_width_set(1.0)

_shader_solid = None

def _create_biased_solid_shader():
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC3', "pos")
    info.fragment_out(0, 'VEC4', "fragColor")
    info.push_constant('MAT4', "ModelViewMatrix")
    info.push_constant('MAT4', "ProjectionMatrix")
    info.push_constant('VEC4', "color")
    info.push_constant('FLOAT', "depth_bias_multiplier")
    info.vertex_source("""
void main() {
    vec4 view_pos = ModelViewMatrix * vec4(pos, 1.0);
    gl_Position = ProjectionMatrix * view_pos;
    
    if (abs(ProjectionMatrix[3][3]) < 0.001) {
        gl_Position.z -= (0.0005 * depth_bias_multiplier) * gl_Position.w;
    } else {
        gl_Position.z += (0.001 * depth_bias_multiplier) * ProjectionMatrix[2][2];
    }
}
""")
    info.fragment_source("""
void main() {
    fragColor = color;
}
""")
    return gpu.shader.create_from_info(info)

def _get_3d_shader(style):
    """Return the appropriate shader for the requested line style."""
    global _shader_solid
    
    if _shader_solid is None:
        _shader_solid = _create_biased_solid_shader()
        
    return _shader_solid

def any_viewport_active():
    """Return True if any viewport has the seams overlay enabled."""
    return any(s.enabled for s in _viewport_states.values())

def invalidate_3d_boundaries():
    """Clear cached 3D boundary data for all objects (forces re-extraction)."""
    from . import draw as _draw
    for cache in _draw._obj_cache.values():
        cache['seam_3d_coords_unsel'] = None
        cache['seam_3d_coords_sel'] = None
        cache['seam_3d_coords_act'] = None
        cache['seam_3d_batch_unsel'] = None
        cache['seam_3d_batch_sel'] = None
        cache['seam_3d_batch_act'] = None

def tag_3d_redraw():
    """Tag all 3D viewports with active overlay for redraw."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and is_active_in_space(space):
                            area.tag_redraw()
                            break
    except Exception:
        pass

def register():
    global _draw_handler_3d
    _draw_handler_3d = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback_3d, (), 'WINDOW', 'POST_VIEW'
    )

def unregister():
    global _draw_handler_3d, _shader_solid
    
    # Restore native seams in all viewports before removing handler
    _restore_all_native_seams()
    
    if _draw_handler_3d is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler_3d, 'WINDOW')
        _draw_handler_3d = None
    
    _viewport_states.clear()
    _shader_solid = None

def _restore_all_native_seams():
    """On addon disable/unregister, restore ALL saved native seam states."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if space.type != 'VIEW_3D':
                        continue
                    ptr = space.as_pointer()
                    state = _viewport_states.get(ptr)
                    if state and state.native_seams_saved:
                        space.overlay.show_edge_seams = state.native_seams_original
                        state.native_seams_saved = False
    except Exception:
        pass
