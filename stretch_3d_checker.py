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
    if (abs(ProjectionMatrix[3][3]) < 0.001) {
        // Perspective
        float dist = length(view_pos.xyz);
        float bias = (dist * 0.0005 + 0.0001);
        vec3 view_dir = view_pos.xyz / dist;
        view_pos.xyz -= view_dir * bias;
        gl_Position = ProjectionMatrix * view_pos;
    } else {
        // Orthographic
        gl_Position = ProjectionMatrix * view_pos;
        gl_Position.z += 0.001 * ProjectionMatrix[2][2];
    }
}
"""

_FRAG_SRC = """
void main()
{
    vec2 fuv = floor(v_uv * divisions);
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
        
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC2', "realUV")
        info.vertex_in(2, 'VEC4', "heatColor")
        
        vert_out = gpu.types.GPUStageInterfaceInfo("stretch_3d_checker_iface")
        vert_out.smooth('VEC2', "v_uv")
        vert_out.smooth('VEC4', "v_heatColor")
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
        
    shader.uniform_float("divisions", divisions)

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
