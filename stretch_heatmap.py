"""
stretch_heatmap.py

Phase 4: Heatmap overlay for UV stretch visualization.
Calculates per-vertex stretch values from the cached Jacobians and outputs a 
smooth color gradient (Blue = compressed, Gray = perfect, Red = stretched).
"""

import gpu
import math
from gpu_extras.batch import batch_for_shader

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
        col_blue = (0.0, 0.05, 1.0, 1.0)
        col_gray = (0.214, 0.214, 0.214, 0.0) # ~0.5 sRGB mid-gray, fully transparent
        col_red  = (1.0, 0.05, 0.0, 1.0)

        for cache in obj_cache.values():
            islands = cache.get('islands')
            if not islands:
                continue
                
            tex_w = cache.get('tex_w', 1024.0)
            tex_h = cache.get('tex_h', 1024.0)
            target_texel = cache.get('target_texel', 500.0)
            
            for isle in islands:
                if target_texel > 0:
                    scale = target_texel / math.sqrt(tex_w * tex_h)
                else:
                    scale = math.sqrt(isle.uv_area / isle.surface_area) if isle.surface_area > 0 else 1.0

                # 1. Area-weighted average of Jacobians per UV vertex
                vert_M_sum = {}
                vert_area_sum = {}

                for i, tri in enumerate(isle.tris):
                    M = isle.jacobians[i]
                    u0, v0 = tri[0]
                    u1, v1 = tri[1]
                    u2, v2 = tri[2]
                    area = abs((u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)) * 0.5

                    for u, v in tri:
                        key = (round(u, 5), round(v, 5))
                        if key not in vert_M_sum:
                            vert_M_sum[key] = [0.0, 0.0, 0.0, 0.0]
                            vert_area_sum[key] = 0.0
                        
                        vert_M_sum[key][0] += M[0] * area
                        vert_M_sum[key][1] += M[1] * area
                        vert_M_sum[key][2] += M[2] * area
                        vert_M_sum[key][3] += M[3] * area
                        vert_area_sum[key] += area

                # 2. Compute stretch color per vertex
                for i, tri in enumerate(isle.tris):
                    for u, v in tri:
                        key = (round(u, 5), round(v, 5))
                        area = vert_area_sum[key]
                        if area > 1e-8:
                            M_avg = [m / area for m in vert_M_sum[key]]
                        else:
                            M_avg = [1.0, 0.0, 0.0, 1.0]

                        M00 = M_avg[0] * scale
                        M01 = M_avg[1] * scale
                        M10 = M_avg[2] * scale
                        M11 = M_avg[3] * scale

                        # Area Stretch: determinant of M (which maps UV length to 3D length)
                        det_M = M00 * M11 - M01 * M10
                        area_stretch = math.sqrt(abs(det_M)) if det_M != 0 else 1.0

                        # Angle Stretch: ratio of singular values
                        trace = M00 + M11
                        diff = M00 - M11
                        desc = math.sqrt(max(0.0, diff*diff + 4.0 * M01 * M10))
                        s1 = (trace + desc) * 0.5
                        s2 = (trace - desc) * 0.5
                        
                        if abs(s1) < 1e-8 or abs(s2) < 1e-8:
                            angle_stretch = 1.0
                        else:
                            angle_stretch = (abs(s1/s2) + abs(s2/s1)) * 0.5

                        # Log2 error mapping
                        area_err = math.log2(area_stretch) if area_stretch > 1e-8 else 0.0
                        angle_err = math.log2(abs(angle_stretch)) if abs(angle_stretch) > 1e-8 else 0.0

                        # Combine
                        weight = 0.5
                        sign = 1.0 if area_err >= 0 else -1.0
                        total_err = sign * (abs(area_err) * (1.0 - weight) + angle_err * weight)

                        # Map to color
                        val = max(-1.0, min(1.0, total_err))
                        if val <= 0:
                            col = _lerp_color(col_gray, col_blue, -val)
                        else:
                            col = _lerp_color(col_gray, col_red, val)

                        coords.append((u, v, 0.0))
                        colors.append(col)

        if not coords:
            return None

        return batch_for_shader(shader, 'TRIS', {"pos": coords, "color": colors})

    except Exception as e:
        import traceback
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
                import traceback
                print(f"[UVO] stretch_heatmap draw error: {e}")
                traceback.print_exc()
                _draw_error_printed = True
