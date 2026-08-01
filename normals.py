import bpy
import gpu
import math
from gpu_extras.batch import batch_for_shader

_batch_lines = None
_batch_tris = None

def get_arrow_geometry(origin, vector, is_positive):
    ox, oy = origin
    vx, vy = vector
    
    length = math.hypot(vx, vy)
    if length < 0.0001:
        return [], [], []
        
    dir_x, dir_y = vx / length, vy / length
    
    end_x = ox + vx
    end_y = oy + vy
    
    line_coords = [ (ox, oy, 0.0), (end_x, end_y, 0.0) ]
    
    head_len = 0.04
    head_half_width = 0.02
    
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

def _add_arrow(origin, vector, is_positive, color, out_lines, out_lines_colors, out_tris, out_tris_colors):
    lc, tc, hl = get_arrow_geometry(origin, vector, is_positive)
    
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


def _rebuild_batches(props):
    global _batch_lines, _batch_tris
    
    line_coords, line_colors = [], []
    tri_coords, tri_colors = [], []
    
    max_len = 0.2  # Max length representing 1.0 component
    diag = max_len * 0.7071  # sin(45) and cos(45)
    
    # Simulate Island 1: Positive X, Positive Y, Positive Z
    origin1 = (0.3, 0.5)
    if props.normal_filter_x:
        _add_arrow(origin1, (max_len * 0.8, 0.0), True, (1.0, 0.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
    if props.normal_filter_y:
        _add_arrow(origin1, (0.0, max_len * 0.5), True, (0.0, 1.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
    if props.normal_filter_z:
        _add_arrow(origin1, (diag * 0.3, diag * 0.3), True, (0.0, 0.5, 1.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
        
    # Simulate Island 2: Negative X, Negative Y, Negative Z
    origin2 = (0.7, 0.5)
    if props.normal_filter_x:
        _add_arrow(origin2, (-max_len * 0.6, 0.0), False, (1.0, 0.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
    if props.normal_filter_y:
        _add_arrow(origin2, (0.0, -max_len * 0.9), False, (0.0, 1.0, 0.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)
    if props.normal_filter_z:
        _add_arrow(origin2, (-diag * 0.7, -diag * 0.7), False, (0.0, 0.5, 1.0, 1.0), line_coords, line_colors, tri_coords, tri_colors)

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

def draw(props, shader, context):
    _rebuild_batches(props)
    
    if _batch_lines:
        gpu.state.line_width_set(2.0)
        _batch_lines.draw(shader)
        
    if _batch_tris:
        _batch_tris.draw(shader)

def clear():
    global _batch_lines, _batch_tris
    _batch_lines = None
    _batch_tris = None
