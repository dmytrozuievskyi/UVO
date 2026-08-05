import bpy
import gpu
import math
from gpu_extras.batch import batch_for_shader

_batch_lines = None
_batch_tris = None

def get_arrow_geometry(origin, vector, is_positive, max_len):
    ox, oy = origin
    vx, vy = vector
    
    length = math.hypot(vx, vy)
    if length < 0.0001:
        return [], [], []
        
    dir_x, dir_y = vx / length, vy / length
    
    end_x = ox + vx
    end_y = oy + vy
    
    line_coords = [ (ox, oy, 0.0), (end_x, end_y, 0.0) ]
    
    # max_len is 45px, so 10px is (12.0/48.0) of max_len
    head_len = max_len * (12.0 / 48.0)
    head_half_width = head_len * 0.5
    
    # Don't scale down the width of the arrowhead so they all look consistent.
    # Just cap the length so it doesn't extend backwards past the origin.
    if head_len > length:
        head_len = length
    
    if is_positive:
        tip_x, tip_y = end_x, end_y
        base_x = tip_x - dir_x * head_len
        base_y = tip_y - dir_y * head_len
        
        perp_x = -dir_y * head_half_width
        perp_y = dir_x * head_half_width
        
        p1 = (tip_x, tip_y, 0.0)
        p2 = (base_x + perp_x, base_y + perp_y, 0.0)
        p3 = (base_x - perp_x, base_y - perp_y, 0.0)
        
        return line_coords, [p1, p2, p3], []
    else:
        tip_x, tip_y = end_x, end_y
        base_x = tip_x + dir_x * head_len
        base_y = tip_y + dir_y * head_len
        
        perp_x = -dir_y * head_half_width
        perp_y = dir_x * head_half_width
        
        p1 = (tip_x, tip_y, 0.0)
        p2 = (base_x + perp_x, base_y + perp_y, 0.0)
        p3 = (base_x - perp_x, base_y - perp_y, 0.0)
        
        hollow_lines = [ p1, p2, p2, p3, p3, p1 ]
        return line_coords, [], hollow_lines

def get_circle_geometry(center, radius, is_filled):
    cx, cy = center
    segs = 16
    pts = []
    for i in range(segs):
        ang = (i / segs) * math.pi * 2
        pts.append((cx + math.cos(ang)*radius, cy + math.sin(ang)*radius, 0.0))
        
    if is_filled:
        tris = []
        c3 = (cx, cy, 0.0)
        for i in range(segs):
            tris.extend([c3, pts[i], pts[(i+1)%segs]])
        return [], tris, []
    else:
        lines = []
        for i in range(segs):
            lines.extend([pts[i], pts[(i+1)%segs]])
        return [], [], lines

def _add_circle(center, radius, is_filled, color, out_lines, out_lines_colors, out_tris, out_tris_colors):
    lc, tc, hl = get_circle_geometry(center, radius, is_filled)
    for _ in range(len(hl) // 2):
        out_lines.extend(hl[-2:])
        out_lines_colors.extend((color, color))
        del hl[-2:]
    for _ in range(len(tc) // 3):
        out_tris.extend(tc[-3:])
        out_tris_colors.extend((color, color, color))
        del tc[-3:]

def _add_arrow(origin, vector, is_positive, color, out_lines, out_lines_colors, out_tris, out_tris_colors, max_len):
    lc, tc, hl = get_arrow_geometry(origin, vector, is_positive, max_len)
    
    for _ in range(len(lc) // 2):
        out_lines.extend(lc[-2:])
        out_lines_colors.extend((color, color))
        del lc[-2:]
        
    for _ in range(len(hl) // 2):
        out_lines.extend(hl[-2:])
        out_lines_colors.extend((color, color))
        del hl[-2:]
        
    for _ in range(len(tc) // 3):
        out_tris.extend(tc[-3:])
        out_tris_colors.extend((color, color, color))
        del tc[-3:]


_normal_data = {}
_cached_filter_state = None

_FILL_VERT_SRC = """
void main() {
    fragColor = color;
    fragType = type;
    fragPos = pos;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_FILL_FRAG_SRC = """
void main() {
    // Generate checkerboard pattern
    float d = float(divisions);
    int iu = int(mod(floor(fragPos.x * d), 2.0));
    int iv = int(mod(floor(fragPos.y * d), 2.0));
    
    // Negative axis is checker (10% or 25%). Positive is solid (25%).
    float checkerAlpha = ((iu + iv) % 2 == 0) ? 0.2 : 0.25;
    float solidAlpha = 0.25;
    
    // Map fragType [-1, 1] to [0, 1] for blending
    float mixFactor = clamp((fragType + 1.0) * 0.5, 0.0, 1.0);
    
    // Smoothly blend between checkerboard and solid based on proximity to axis type
    float alpha = mix(checkerAlpha, solidAlpha, mixFactor);
    
    outColor = vec4(fragColor.rgb, alpha);
}
"""

_fill_shader = None

def _get_fill_shader():
    global _fill_shader
    import gpu
    if _fill_shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewProjectionMatrix")
        info.push_constant('FLOAT', "divisions")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC4', "color")
        info.vertex_in(2, 'FLOAT', "type")
        vert_out = gpu.types.GPUStageInterfaceInfo("normals_fill_iface")
        vert_out.smooth('VEC4', "fragColor")
        vert_out.smooth('FLOAT', "fragType")
        vert_out.smooth('VEC3', "fragPos")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "outColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_FILL_FRAG_SRC)
        _fill_shader = gpu.shader.create_from_info(info)
    return _fill_shader


def rebuild_from_worker_data(results):
    global _normal_data, _cached_filter_state
    _normal_data = results
    _cached_filter_state = None  # Force rebuild on next draw

_cached_zoom = 0.0

def _rebuild_batches(props, context):
    global _batch_lines, _batch_tris, _batch_fill, _normal_data, _cached_filter_state, _cached_zoom
    
    from . import stretch_checker
    zoom = stretch_checker.get_zoom(context)
    
    current_filters = (props.normal_filter_x, props.normal_filter_y, props.normal_filter_z)
    if _cached_filter_state == current_filters and abs(_cached_zoom - zoom) < (zoom * 0.01):
        return
        
    _cached_filter_state = current_filters
    _cached_zoom = zoom
    
    line_coords, line_colors = [], []
    tri_coords, tri_colors = [], []
    fill_coords, fill_colors, fill_types = [], [], []
    
    # Target size: roughly 30 pixels on screen.
    # zoom = pixels_per_uv / 256.0
    # pixels_per_uv = zoom * 256.0
    # max_len in UV units = target_pixels / pixels_per_uv
    pixels_per_uv = zoom * 256.0
    max_len = 45.0 / max(1.0, pixels_per_uv)
    
    diag = max_len * 0.7071  # sin(45) and cos(45)
    
    circle_radius = max_len * (10.0 / 45.0)  # 10px radius = 20px diameter
    
    
    min_island_px_area = 8100.0
    ppuv_sq = pixels_per_uv * pixels_per_uv
    
    for obj_name, obj_results in _normal_data.items():
        for res in obj_results:
            island_groups = res['groups']
            if not island_groups:
                continue
                
            if res.get('fill_coords'):
                fill_coords.extend(res['fill_coords'])
                fill_colors.extend(res['fill_colors'])
                fill_types.extend(res.get('fill_types', [1.0] * len(res['fill_coords'])))
                
            island_uv_area = island_groups[0].get('island_uv_area', 1.0)
            if island_uv_area * ppuv_sq < min_island_px_area:
                continue
                
                
            for g in island_groups:
                center = g['center']
                nx, ny, nz = g['normal']
                px_u, px_v = g['proj_x']
                py_u, py_v = g['proj_y']
                pz_u, pz_v = g['proj_z']
                
                abs_n = [abs(nx), abs(ny), abs(nz)]
                max_idx = abs_n.index(max(abs_n))
                
                if props.normal_filter_x:
                    if max_idx == 0:
                        _add_circle(center, circle_radius, nx > 0, (1.0, 0.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
                    if abs(px_u) > 1e-4 or abs(px_v) > 1e-4:
                        _add_arrow(center, (px_u * max_len, px_v * max_len), True, (1.0, 0.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors, max_len)
                        
                if props.normal_filter_y:
                    if max_idx == 1:
                        _add_circle(center, circle_radius, ny > 0, (0.0, 1.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
                    if abs(py_u) > 1e-4 or abs(py_v) > 1e-4:
                        _add_arrow(center, (py_u * max_len, py_v * max_len), True, (0.0, 1.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors, max_len)
                        
                if props.normal_filter_z:
                    if max_idx == 2:
                        _add_circle(center, circle_radius, nz > 0, (0.0, 0.5, 1.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
                    if abs(pz_u) > 1e-4 or abs(pz_v) > 1e-4:
                        _add_arrow(center, (pz_u * max_len, pz_v * max_len), True, (0.0, 0.5, 1.0, 1.0), line_coords, line_colors, tri_coords, tri_colors, max_len)

    try:
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    except Exception:
        shader = None

    if shader and line_coords:
        _batch_lines = batch_for_shader(shader, 'LINES', {"pos": line_coords, "color": line_colors})
    else:
        _batch_lines = None
        
    if shader and tri_coords:
        _batch_tris = batch_for_shader(shader, 'TRIS', {"pos": tri_coords, "color": tri_colors})
    else:
        _batch_tris = None
        
    if fill_coords:
        fill_shader = _get_fill_shader()
        if fill_shader:
            _batch_fill = batch_for_shader(fill_shader, 'TRIS', {
                "pos": fill_coords,
                "color": fill_colors,
                "type": fill_types
            })
        else:
            _batch_fill = None
    else:
        _batch_fill = None

def draw(props, shader, context):
    _rebuild_batches(props, context)
    
    if _batch_fill:
        fill_shader = _get_fill_shader()
        if fill_shader:
            fill_shader.bind()
            
            from . import stretch_checker
            z_lvl = stretch_checker.get_zoom_level(context)
            divs = stretch_checker.get_divisions(z_lvl)
            fill_shader.uniform_float("divisions", float(divs))
            
            _batch_fill.draw(fill_shader)
            shader.bind()
    
    if _batch_lines:
        gpu.state.line_width_set(2.0)
        _batch_lines.draw(shader)
        
    if _batch_tris:
        _batch_tris.draw(shader)

def clear():
    global _batch_lines, _batch_tris, _batch_fill, _normal_data, _cached_filter_state
    _batch_lines = None
    _batch_tris = None
    _batch_fill = None
    _normal_data = {}
    _cached_filter_state = None
