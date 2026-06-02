import gpu
import math
import traceback
from gpu_extras.batch import batch_for_shader
from . import stretch

_VERT_SRC = """
void main() {
    fragColor = color;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_FRAG_SRC = """
void main() {
    float a = fragColor.a;
    if (transparent_gray == 0) {
        a = 1.0;
    }
    outColor = vec4(fragColor.rgb, a * opacity);
}
"""

_shader = None

def _get_shader():
    global _shader
    if _shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewProjectionMatrix")
        info.push_constant('FLOAT', "opacity")
        info.push_constant('INT', "transparent_gray")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC4', "color")
        vert_out = gpu.types.GPUStageInterfaceInfo("stretch_heatmap_iface")
        vert_out.smooth('VEC4', "fragColor")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "outColor")
        info.vertex_source(_VERT_SRC)
        info.fragment_source(_FRAG_SRC)
        _shader = gpu.shader.create_from_info(info)
    return _shader


def clear():
    global _shader
    _shader = None


def build_geometry_batch(obj_cache, props):
    try:
        shader = _get_shader()
        if shader is None:
            print("[UVO] Heatmap shader failed to compile.")
            return None

        coords = []
        colors = []



        for cache in obj_cache.values():
            islands = cache.get('islands')
            if not islands:
                continue

            tex_w = cache.get('tex_w', 1024.0)
            tex_h = cache.get('tex_h', 1024.0)
            target_texel = cache.get('target_texel', 0.0)

            for isle in islands:
                scale, scale_u, scale_v = stretch.compute_scale_factors(isle, tex_w, tex_h, target_texel)

                vert_M_sum, vert_area_sum = stretch.compute_vertex_jacobians(isle)

                heat_colors = {}
                for key, area in vert_area_sum.items():
                    if area > 1e-8:
                        M_avg = [m / area for m in vert_M_sum[key]]
                    else:
                        M_avg = [1.0, 0.0, 0.0, 1.0]

                    area_err, angle_err = stretch.compute_stretch_metrics(M_avg, scale_u, scale_v)
                    heat_colors[key] = stretch.error_to_color(area_err, angle_err, 'heatmap')


                for tri in isle.tris:
                    for u, v in tri:
                        key = (round(u, 5), round(v, 5))
                        coords.append((u, v, 0.0))
                        colors.append(heat_colors.get(key, stretch.COL_GRAY))

        if not coords:
            return False

        return batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})

    except Exception as e:
        print(f"[UVO] stretch_heatmap build_geometry_batch error: {e}")
        traceback.print_exc()
        return False


_draw_error_printed = False

def draw(batch, opacity, transparent_gray=False):
    global _draw_error_printed
    shader = _get_shader()
    if shader and batch:
        try:
            shader.bind()
            shader.uniform_float("opacity", opacity)
            shader.uniform_int("transparent_gray", 1 if transparent_gray else 0)
            batch.draw(shader)
        except Exception as e:
            if not _draw_error_printed:
                print(f"[UVO] stretch_heatmap draw error: {e}")
                traceback.print_exc()
                _draw_error_printed = True


def build_batch_from_precomputed(coords, colors):
    """Build GPU batch from pre-computed worker data (no math, just batch_for_shader)."""
    shader = _get_shader()
    if shader is None or not coords:
        return False
    try:
        return batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})
    except Exception as e:
        print(f"[UVO] stretch_heatmap batch_from_precomputed error: {e}")
        traceback.print_exc()
        return False
