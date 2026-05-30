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


def _lerp_color(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
        c1[3] + (c2[3] - c1[3]) * t
    )


def build_geometry_batch(obj_cache, props):
    try:
        shader = _get_shader()
        if shader is None:
            print("[UVO] Heatmap shader failed to compile.")
            return None

        coords = []
        colors = []

        # Linear color space
        col_blue = (0.0, 0.0, 1.0, 1.0)
        col_gray = (0.214, 0.214, 0.214, 0.0)
        col_red  = (1.0, 0.0, 0.0, 1.0)

        for cache in obj_cache.values():
            islands = cache.get('islands')
            if not islands:
                continue

            tex_w = cache.get('tex_w', 1024.0)
            tex_h = cache.get('tex_h', 1024.0)
            target_texel = cache.get('target_texel', 500.0)

            for isle in islands:
                if target_texel > 0:
                    scale_u = target_texel / tex_w
                    scale_v = target_texel / tex_h
                else:
                    scale = math.sqrt(isle.uv_area / isle.surface_area) if isle.surface_area > 0 else 1.0
                    aspect = tex_h / tex_w if tex_w > 0 else 1.0
                    scale_u = scale * math.sqrt(aspect)
                    scale_v = scale / math.sqrt(aspect)

                # 1. Area-weighted average of Jacobians per UV vertex
                vert_M_sum, vert_area_sum = stretch.compute_vertex_jacobians(isle)

                # 2. Pre-compute heat color per unique vertex
                heat_colors = {}
                for key, area in vert_area_sum.items():
                    if area > 1e-8:
                        M_avg = [m / area for m in vert_M_sum[key]]
                    else:
                        M_avg = [1.0, 0.0, 0.0, 1.0]

                    M00 = M_avg[0] * scale_u
                    M01 = M_avg[1] * scale_v
                    M10 = M_avg[2] * scale_u
                    M11 = M_avg[3] * scale_v

                    # Area Stretch: determinant of M (which maps UV length to 3D length)
                    det_M = M00 * M11 - M01 * M10
                    area_stretch = math.sqrt(abs(det_M)) if det_M != 0 else 1.0

                    # Angle Stretch: ratio of singular values
                    E = (M00 + M11) * 0.5
                    F = (M00 - M11) * 0.5
                    G = (M10 + M01) * 0.5
                    H = (M10 - M01) * 0.5
                    Q = math.sqrt(E*E + H*H)
                    R = math.sqrt(F*F + G*G)
                    s1 = Q + R
                    s2 = abs(Q - R)

                    if abs(s1) < 1e-8 or abs(s2) < 1e-8:
                        angle_stretch = 1.0
                    else:
                        angle_stretch = (abs(s1/s2) + abs(s2/s1)) * 0.5

                    area_err = math.log2(area_stretch) if area_stretch > 1e-8 else 0.0
                    angle_err = math.log2(abs(angle_stretch)) if abs(angle_stretch) > 1e-8 else 0.0

                    # Weighted sum of area + angle error, signed by area direction
                    weight = 0.5
                    sign = 1.0 if area_err >= 0 else -1.0
                    total_err = sign * (abs(area_err) * (1.0 - weight) + angle_err * weight)

                    val = max(-1.0, min(1.0, total_err))

                    if val <= 0:
                        heat_colors[key] = _lerp_color(col_gray, col_blue, -val)
                    else:
                        heat_colors[key] = _lerp_color(col_gray, col_red, val)

                # 3. Emit tri vertices with pre-computed colors
                for tri in isle.tris:
                    for u, v in tri:
                        key = (round(u, 5), round(v, 5))
                        coords.append((u, v, 0.0))
                        colors.append(heat_colors.get(key, col_gray))

        if not coords:
            return None

        return batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})

    except Exception as e:
        print(f"[UVO] stretch_heatmap build_geometry_batch error: {e}")
        traceback.print_exc()
        return None


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
        return None
    try:
        return batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})
    except Exception as e:
        print(f"[UVO] stretch_heatmap batch_from_precomputed error: {e}")
        traceback.print_exc()
        return None
