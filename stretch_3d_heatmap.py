import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_shader = None

_VERT_SRC = """
void main()
{
    fcolor = color;
    vec4 clip_pos = ModelViewProjectionMatrix * vec4(pos, 1.0);
    clip_pos.z -= 0.0005 * clip_pos.w; // depth bias to prevent z-fighting
    gl_Position = clip_pos;
}
"""

_FRAG_SRC = """
void main()
{
    fragColor = vec4(fcolor.rgb, 1.0 * opacity);
}
"""

def get_shader():
    global _shader
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewProjectionMatrix")
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
    shader = get_shader()
    shader.bind()
    shader.uniform_float("opacity", opacity)

    base_mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()

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
            mvp = base_mvp @ obj.matrix_world
            shader.uniform_float("ModelViewProjectionMatrix", mvp)
            cache['batch'].draw(shader)
