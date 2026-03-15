# UVO — UV Overlays for Blender

A Blender addon that adds a suite of dynamic, real-time overlays to the UV Editor, helping you catch UV problems before they reach your textures.

---

## Overlays

### ID Overlay
Assigns a distinct, stable color to each UV island or mesh component so you can immediately see how your UVs are organized.

- **Object mode** — one color per object
- **Island mode** — one color per topologically connected UV island
- Adjustable opacity

---

### Intersect Overlay
Highlights overlapping and stacked UV islands in real-time.

- Detects overlaps **within** a single object and **across** all objects in Edit Mode simultaneously
- **Tiled mode** — folds all UV tiles into (0,1) before detection, finding overlaps between islands in different tiles that share the same texel space. Use for tiling or repeating textures
- **UDIM mode** — each tile is treated independently. Use when each tile bakes to a separate texture
- Hatching on intersecting islands, cross-hatching on perfectly stacked islands, red fill on the overlapping area
- Adjustable opacity

---

### Padding Overlay
Visualizes the padding zone around each island and flags islands whose zones overlap, indicating potential mipmap bleed.

- Set texture resolution (256 – 8192, width and height independently)
- Set padding size in pixels (2 – 32)
- Green outline = safe, red outline = padding violation
- Aspect-corrected display for non-square textures

---

## Installation

1. Download the latest release `.zip` from the [Releases](../../releases) page
2. In Blender, go to **Edit → Preferences → Add-ons → Install from Disk**
3. Select the downloaded `.zip`
4. Enable **UVO - UV Overlays** in the addon list

---

## Usage

1. Enter **Edit Mode** on a mesh
2. Open the **UV Editor**
3. Make sure **Overlays** are enabled in the UV Editor header
4. Click the **UVO button** in the tool header (right side) to open the overlay panel
5. Enable any combination of ID, Intersect, and Padding overlays

**Live Update** — when enabled, overlays refresh on every UV change. Disable for smoother editing on complex meshes; overlays will update once after each action completes.

---

## Compatibility

Requires Blender 4.2 or newer.

---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE)
