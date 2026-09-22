import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_shader = None

_VERT_SRC = """
void main()
{
    fcolor = color;
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
    fragColor = vec4(fcolor.rgb, 1.0 * opacity);
}
"""

def _get_shader():
    global _shader
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewMatrix")
        info.push_constant('MAT4',  "ProjectionMatrix")
        info.push_constant('FLOAT', "opacity")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC4', "color")
        vert_out = gpu.types.GPUStageInterfaceInfo("stretch_3d_heatmap_iface")
        vert_out.smooth('VEC4', "fcolor")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source(_VERT_SRC)
        info.fragment_source(_FRAG_SRC)
        _shader = gpu.shader.create_from_info(info)
    return _shader


def draw(stretch_3d_cache, opacity):
    import bpy
    shader = _get_shader()
    shader.bind()
    shader.uniform_float("opacity", opacity)

    base_mv = gpu.matrix.get_model_view_matrix()
    base_proj = gpu.matrix.get_projection_matrix()

    for obj_name, cache in stretch_3d_cache.items():
        if cache.get('batch') is None:
            if not cache.get('world_coords') or not cache.get('heatmap_colors'):
                continue
            cache['batch'] = batch_for_shader(
                shader, 'TRIS',
                {"pos": cache['world_coords'], "color": cache['heatmap_colors']}
            )
        
        obj = bpy.data.objects.get(obj_name)
        if obj:
            shader.uniform_float("ModelViewMatrix", base_mv)
            shader.uniform_float("ProjectionMatrix", base_proj)
            cache['batch'].draw(shader)
