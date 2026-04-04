import colorsys

_HUE_MIN  = 0.06   # avoid red at both wheel ends — conflicts with error overlays
_HUE_MAX  = 0.94
_HUE_SPAN = _HUE_MAX - _HUE_MIN

_SAT_EVEN = 0.85
_SAT_ODD  = 0.55
_VAL      = 0.90


def get_distinct_color(index, total, seed_offset=0.0, alpha=0.5):
    """Return RGBA for island/object `index` out of `total`.

    Hue slots are interleaved (evens take the first half, odds the second)
    so consecutive islands are always ~half the wheel apart, preventing
    adjacent similar colours. Saturation alternates to double capacity.
    """
    n = max(total, 1)
    half = (n + 1) // 2
    slot = (index // 2) if (index % 2 == 0) else (half + index // 2)

    h_norm = (slot / n + seed_offset) % 1.0
    h = _HUE_MIN + h_norm * _HUE_SPAN
    s = _SAT_EVEN if (index % 2 == 0) else _SAT_ODD

    r, g, b = colorsys.hsv_to_rgb(h, s, _VAL)
    return (r, g, b, alpha)


def get_string_hash(s):
    # 24-bit polynomial rolling hash → ~16M slots, negligible collisions.
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    return (h & 0xFFFFFF) / 0x1000000


_last: dict = {}



def _debug_enabled() -> bool:
    try:
        import bpy as _bpy
        prefs = _bpy.context.preferences.addons.get(__package__)
        if prefs is None:
            return False
        return getattr(prefs.preferences, 'debug', False)
    except Exception:
        # Running outside Blender (worker subprocess) — always log.
        return True


def log(key: str, msg: str) -> None:
    """Print [UVO] <key>: <msg> only when debug is on AND msg changed."""
    if not _debug_enabled():
        return
    if _last.get(key) == msg:
        return
    _last[key] = msg
    print(f"[UVO] {key}: {msg}")


def log_clear() -> None:
    """Reset dedup state on unregister."""
    _last.clear()