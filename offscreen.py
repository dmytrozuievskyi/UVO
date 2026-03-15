import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from . import utils

_offscreen        = None    # gpu.types.GPUOffScreen
_offscreen_size   = (0, 0)
_offscreen_dirty  = True
_offscreen_batch  = None    # (verts, uvs, idxs) raw quad data
_offscreen_comp_batch = None  # compiled GPUBatch for composite quad (cached)
_offscreen_shader = None    # compiled compositing shader
_last_view_matrix = None    # snapshot for pan/zoom detection


def _get_shader():
    # Additive composite: discards pixels below threshold (one island),
    # outputs red for pixels >= threshold (two+ islands overlapping).
    global _offscreen_shader
    if _offscreen_shader is not None:
        return _offscreen_shader
    try:
        iface = gpu.types.GPUStageInterfaceInfo("uvo_iface")
        iface.smooth('VEC2', "uvInterp")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', "pos")
        info.vertex_in(1, 'VEC2', "uv")
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.sampler(0, 'FLOAT_2D', "image")
        info.push_constant('FLOAT', "opacity")
        info.push_constant('FLOAT', "threshold")
        info.vertex_source("""
void main() {
    uvInterp = uv;
    gl_Position = vec4(pos, 0.0, 1.0);
}
""")
        info.fragment_source("""
void main() {
    float val = texture(image, uvInterp).r;
    if (val < threshold) discard;
    fragColor = vec4(1.0, 0.0, 0.0, opacity);
}
""")
        _offscreen_shader = gpu.shader.create_from_info(info)
        utils.log("shader", "offscreen shader created OK")
    except Exception as e:
        utils.log("shader", f"shader creation failed: {e}")
        _offscreen_shader = None
    return _offscreen_shader


def _ensure(width, height):
    global _offscreen, _offscreen_size, _offscreen_batch, _offscreen_dirty
    global _offscreen_comp_batch
    if _offscreen is not None and _offscreen_size == (width, height):
        return
    if _offscreen is not None:
        try:
            _offscreen.free()
        except Exception:
            pass
    _offscreen            = gpu.types.GPUOffScreen(width, height)
    _offscreen_size       = (width, height)
    _offscreen_dirty      = True
    _offscreen_comp_batch = None
    # Fullscreen NDC quad [-1,1] → UV [0,1]
    verts = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    uvs   = [(0.0, 0.0),   (1.0, 0.0),  (1.0, 1.0), (0.0, 1.0)]
    idxs  = [(0, 1, 2), (0, 2, 3)]
    _offscreen_batch = (verts, uvs, idxs)


def mark_dirty():
    global _offscreen_dirty
    _offscreen_dirty = True


def check_view_matrix():
    """Returns True if the UV editor view has changed (pan/zoom) since last call."""
    global _last_view_matrix
    cur = tuple(tuple(row) for row in gpu.matrix.get_projection_matrix())
    if cur != _last_view_matrix:
        _last_view_matrix = cur
        return True
    return False


def render(tris, base_shader, gray_value=0.5):
    """Draw intersecting island tris into the offscreen buffer at gray_value brightness.

    Uses ADDITIVE blend — pixels covered by N islands accumulate to N*gray_value.
    composite() threshold is gray_value*1.5, so 2+ islands exceed it → red fill.
    """
    global _offscreen_dirty

    region = bpy.context.region
    if region:
        _ensure(region.width, region.height)

    if not _offscreen or not tris:
        _offscreen_dirty = False
        return

    if not _offscreen_dirty:
        return

    utils.log("offscreen", f"render: tris={len(tris)}, size={_offscreen_size}")

    coords = []
    colors = []
    for tri in tris:
        for v in tri:
            coords.append((v[0], v[1], 0.0))
            colors.append((gray_value, gray_value, gray_value, 1.0))

    with _offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.blend_set('ADDITIVE')
        gpu.state.depth_test_set('NONE')
        if coords:
            batch = batch_for_shader(base_shader, 'TRIS', {"pos": coords, "color": colors})
            base_shader.bind()
            batch.draw(base_shader)
        gpu.state.blend_set('NONE')

    _offscreen_dirty = False


def composite(opacity, threshold=0.6):
    """Composite offscreen texture over the viewport — red where pixels exceed threshold."""
    if not _offscreen or not _offscreen_batch:
        return False

    sh = _get_shader()
    if sh is None:
        utils.log("shader", "unavailable — overlap fill skipped")
        return False

    global _offscreen_comp_batch
    if _offscreen_comp_batch is None:
        verts, uvs, idxs = _offscreen_batch
        _offscreen_comp_batch = batch_for_shader(
            sh, 'TRIS', {"pos": verts, "uv": uvs}, indices=idxs
        )
    gpu.state.blend_set('ALPHA')
    sh.bind()
    sh.uniform_sampler("image", _offscreen.texture_color)
    sh.uniform_float("opacity", opacity)
    sh.uniform_float("threshold", threshold)
    _offscreen_comp_batch.draw(sh)
    return True


def free():
    """Release all GPU resources. Call on addon unregister."""
    global _offscreen, _offscreen_size, _offscreen_dirty
    global _offscreen_batch, _offscreen_comp_batch, _offscreen_shader, _last_view_matrix
    if _offscreen is not None:
        try:
            _offscreen.free()
        except Exception:
            pass
    _offscreen            = None
    _offscreen_size       = (0, 0)
    _offscreen_dirty      = True
    _offscreen_batch      = None
    _offscreen_comp_batch = None
    _offscreen_shader     = None
    _last_view_matrix     = None