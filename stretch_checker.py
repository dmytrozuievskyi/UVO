import gpu
import math
import traceback
from gpu_extras.batch import batch_for_shader
from . import stretch

_ZOOM_THRESHOLDS = [4.0, 8.0, 16.0, 32.0]
_ZOOM_DIVISIONS  = [10, 20, 40, 80, 160]


def get_zoom(context):
    """Zoom factor relative to 256 px/UV unit, derived from projection matrix."""
    try:
        import gpu
        matrix = gpu.matrix.get_projection_matrix()
        pixels_per_uv = abs(matrix[0][0]) * context.region.width * 0.5
        return max(0.01, pixels_per_uv / 256.0)
    except Exception:
        return 1.0


def get_zoom_level(context):
    z = get_zoom(context)
    for i, threshold in enumerate(_ZOOM_THRESHOLDS):
        if z < threshold:
            return i + 1
    return 5


def get_divisions(zoom_level):
    return _ZOOM_DIVISIONS[max(0, min(zoom_level - 1, 4))]



_VERT_SRC = """
void main() {
    uvCoord = warpedUV;
    heatColor = color;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

# Per-fragment checker. mod() avoids negative-modulo issues outside 0-1 tile.
_FRAG_SRC = """
void main() {
    float d  = float(divisions);
    int iu   = int(mod(floor(uvCoord.x * d), 2.0));
    int iv   = int(mod(floor(uvCoord.y * d), 2.0));
    int cell = (iu + iv) % 2;

    vec3 colDark  = vec3(0.0453, 0.0453, 0.0453);
    vec3 colLight = vec3(0.1008, 0.1008, 0.1008);
    vec3 col = (cell == 1) ? colLight : colDark;

    if (use_tint == 1) {
        float tint = heatColor.a;  // deviation magnitude
        // Dark cells get dimmer tint.
        vec3 targetColor = (cell == 1) ? heatColor.rgb : (heatColor.rgb * 0.5);
        float mixFactor = min(tint * 1.5, 0.75);
        col = mix(col, targetColor, mixFactor);
    }
    float alpha = opacity;

    fragColor = vec4(col, alpha);
}
"""

_shader = None

def _get_shader():
    global _shader
    if _shader is not None:
        return _shader

    try:
        vert_out = gpu.types.GPUStageInterfaceInfo("stretch_checker_iface")
        vert_out.smooth('VEC2', "uvCoord")
        vert_out.smooth('VEC4', "heatColor")

        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4',  "ModelViewProjectionMatrix")
        info.push_constant('FLOAT', "opacity")
        info.push_constant('INT',   "divisions")
        info.push_constant('INT',   "use_tint")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_in(1, 'VEC2', "warpedUV")
        info.vertex_in(2, 'VEC4', "color")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source(_VERT_SRC)
        info.fragment_source(_FRAG_SRC)

        _shader = gpu.shader.create_from_info(info)

    except Exception as e:
        print(f"[UVO] stretch_checker shader compile error: {e}")
        _shader = None

    return _shader


def build_geometry_batch(obj_cache, props):
    """Build a position-only TRIS batch from all island triangles."""
    if not obj_cache:
        return None

    shader = _get_shader()
    if shader is None:
        return None

    coords = []
    warped_uvs = []
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
                heat_colors[key] = stretch.error_to_color(area_err, angle_err, 'checker')

            w_dict = stretch.compute_warped_uvs(isle, vert_M_sum, vert_area_sum, scale)


            for i, tri in enumerate(isle.tris):
                for u, v in tri:
                    key = (round(u, 5), round(v, 5))

                    if key in w_dict:
                        w_u, w_v = w_dict[key]
                    else:
                        w_u, w_v = u, v
                        
                    coords.append((u, v, 0.0))
                    warped_uvs.append((w_u, w_v))
                    colors.append(heat_colors.get(key, stretch.COL_GRAY))

    if not coords:
        return False

    try:
        return batch_for_shader(shader, 'TRIS', {
            "pos": coords,
            "warpedUV": warped_uvs,
            "color": colors
        })
    except Exception as e:
        print(f"[UVO] stretch_checker batch error: {e}")
        traceback.print_exc()
        return None


_draw_error_printed = False

def draw(batch, opacity, context, use_tint=False):
    """Draw the checker grid."""
    global _draw_error_printed
    if batch is None:
        return

    shader = _get_shader()
    if shader is None:
        return

    zoom_level = get_zoom_level(context)
    divisions  = get_divisions(zoom_level)

    try:
        shader.bind()
        shader.uniform_float("opacity",    opacity)
        shader.uniform_int(  "divisions",  divisions)
        shader.uniform_int(  "use_tint",   1 if use_tint else 0)
        batch.draw(shader)
    except Exception as e:
        if not _draw_error_printed:
            import traceback
            print(f"[UVO] stretch_checker draw error: {e}")
            traceback.print_exc()
            _draw_error_printed = True


def build_batch_from_precomputed(coords, warped_uvs, colors):
    """Build GPU batch from pre-computed worker data (no math, just batch_for_shader)."""
    shader = _get_shader()
    if shader is None or not coords:
        return False
    try:
        return batch_for_shader(shader, 'TRIS', {
            "pos": coords,
            "warpedUV": warped_uvs,
            "color": colors,
        })
    except Exception as e:
        print(f"[UVO] stretch_checker batch_from_precomputed error: {e}")
        traceback.print_exc()
        return None


def clear():
    """Release the cached shader on unregister."""
    global _shader
    _shader = None
