import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_shader = None

_VERT_SRC = """
void main()
{
    v_uv = realUV;
    v_heatColor = heatColor;
    vec4 view_pos = ModelViewMatrix * vec4(pos, 1.0);
    
    bool is_persp = abs(ProjectionMatrix[3][3]) < 0.001;
    
    if (is_persp) {
        v_dist = u_view_distance;
        
        // Depth bias must still use the actual per-vertex distance to prevent z-fighting
        float actual_dist = length(view_pos.xyz);
        float bias = (actual_dist * 0.0006 + 0.0005);
        vec3 view_dir = view_pos.xyz / actual_dist;
        view_pos.xyz -= view_dir * bias;
        gl_Position = ProjectionMatrix * view_pos;
    } else {
        v_dist = 1.0 / abs(ProjectionMatrix[0][0]);
        gl_Position = ProjectionMatrix * view_pos;
        gl_Position.z += 0.001 * ProjectionMatrix[2][2];
    }
}
"""

_FRAG_SRC = """
void main()
{
    // Dynamically scale divisions based on distance so squares stay visually consistent
    float scale = exp2(floor(log2(max(v_dist, 0.001))));
    float current_divisions = divisions / scale;
    
    vec2 fuv = floor(v_uv * current_divisions);
    float checker = mod(fuv.x + fuv.y, 2.0);
    int cell = (checker > 0.5) ? 1 : 0;
    
    vec3 col1 = vec3(0.5);
    vec3 col2 = vec3(0.3);
    vec3 col = mix(col2, col1, checker);
    
    if (use_tint == 1.0) {
        float tint = v_heatColor.a;
        vec3 targetColor = (cell == 1) ? v_heatColor.rgb : (v_heatColor.rgb * 0.5);
        float mixFactor = min(tint * 1.5, 0.75);
        col = mix(col, targetColor, mixFactor);
    }
    
    fragColor = vec4(col, opacity);
}
"""

def get_shader():
    global _shader
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewMatrix")
        info.push_constant('MAT4',  "ProjectionMatrix")
        info.push_constant('FLOAT', "opacity")
        info.push_constant('FLOAT', "divisions")
        info.push_constant('FLOAT', "use_tint")
        info.push_constant('FLOAT', "u_view_distance")
        
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC2', "realUV")
        info.vertex_in(2, 'VEC4', "heatColor")
        
        vert_out = gpu.types.GPUStageInterfaceInfo("stretch_3d_checker_iface")
        vert_out.smooth('VEC2', "v_uv")
        vert_out.smooth('VEC4', "v_heatColor")
        vert_out.smooth('FLOAT', "v_dist")
        info.vertex_out(vert_out)

        
        info.fragment_out(0, 'VEC4', "fragColor")
        
        info.vertex_source(_VERT_SRC)
        info.fragment_source(_FRAG_SRC)
        _shader = gpu.shader.create_from_info(info)
    return _shader


def draw(stretch_3d_cache, opacity, context, use_tint=False):
    shader = get_shader()
    shader.bind()
    shader.uniform_float("opacity", opacity)
    shader.uniform_float("use_tint", 1.0 if use_tint else 0.0)
    
    # We use a fixed scale since the mesh doesn't have a "zoom" factor like 2D views
    # but we could link it to the texture resolution or user target texel.
    # For now, 10.0 divisions looks decent.
    divisions = 10.0
    if context.active_object and hasattr(context.active_object, 'uv_id_props'):
        divisions = float(context.active_object.uv_id_props.tex_res_x) / 100.0
        divisions = max(2.0, divisions)
        
    # Scale based on user request (5x larger visual size -> 0.4 multiplier since the old shader had a * 2.0 multiplier)
    # The previous effective baseline was ~20 divisions. 5x larger cells = 4 divisions.
    base_multiplier = 0.4
    
    # Compensate for viewport resolution & UI scale so physical size is stable across monitors
    ui_scale = 1.0
    if hasattr(context, 'preferences'):
        if hasattr(context.preferences, 'system') and hasattr(context.preferences.system, 'ui_scale'):
            ui_scale = context.preferences.system.ui_scale
        elif hasattr(context.preferences, 'view') and hasattr(context.preferences.view, 'ui_scale'):
            ui_scale = context.preferences.view.ui_scale
            
    viewport_height = max(1.0, float(context.region.height))
    
    # This formula ensures that squares take up proportionally more pixels on higher res or higher UI scale.
    res_factor = (viewport_height / 1080.0) / max(0.1, ui_scale)
    
    divisions = divisions * base_multiplier * res_factor
        
    shader.uniform_float("divisions", divisions)
    
    u_view_distance = 10.0
    if hasattr(context, 'region_data') and hasattr(context.region_data, 'view_distance'):
        u_view_distance = float(context.region_data.view_distance)
    shader.uniform_float("u_view_distance", u_view_distance)

    base_mv = gpu.matrix.get_model_view_matrix()
    base_proj = gpu.matrix.get_projection_matrix()

    for obj_name, cache in stretch_3d_cache.items():
        batch_key = 'batch_checker'
        if cache.get(batch_key) is None:
            if not cache.get('world_coords') or not cache.get('uv_coords') or not cache.get('heatmap_colors'):
                continue
            cache[batch_key] = batch_for_shader(
                shader, 'TRIS',
                {
                    "pos": cache['world_coords'], 
                    "realUV": cache['uv_coords'],
                    "heatColor": cache['heatmap_colors']
                }
            )
            
        obj = context.scene.objects.get(obj_name)
        if obj:
            shader.uniform_float("ModelViewMatrix", base_mv)
            shader.uniform_float("ProjectionMatrix", base_proj)
            cache[batch_key].draw(shader)
