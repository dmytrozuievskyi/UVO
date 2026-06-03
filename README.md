# UVO — UV Overlays for Blender

UVO adds a suite of dynamic, real-time overlays to Blender's UV Editor to help you catch mapping issues before they reach your textures.

## Features

### ID Overlay
Color-codes your UVs so you can immediately see how they are organized.
- **Object & Island Modes** — Assigns a distinct color per object, or per topologically connected UV island.

### Intersect Overlay
Highlights overlapping and stacked UV islands in real-time.
- Detects overlaps **within** a single object and **across** all objects in Edit Mode simultaneously.
- **Tiled mode** — Folds all UV tiles into (0,1) before detection, finding overlaps between islands in different tiles that share the same texel space.
- **UDIM mode** — Each tile is treated independently.
- Hatching on intersecting islands, cross-hatching on perfectly stacked islands, and red fill on the overlapping area.

### Padding Overlay
Visualizes the padding zone around each island and flags potential mipmap bleed.

### Stretch Overlay
Visualizes texel density deviation from a target value using a warped checker grid and color heatmap.
- **Checker** — Grid deforms to show UV distortion; uniform squares = no stretch.
- **Heatmap** — Blue (compressed) → gray (uniform) → red (stretched) gradient.
- **Both** — Checker grid with heatmap colors blended into each cell, combining distortion shape and density feedback in a single view.

### Additional Features
- Per-object texture resolution and texel density settings.
- Eyedropper tool to sample density from selected UV islands.
- Correct visualization for non-square textures.

## Installation

1. Download the latest release `.zip` from the [Releases](../../releases) page
2. In Blender, go to **Edit → Preferences → Add-ons → Install from Disk**
3. Select the downloaded `.zip`
4. Enable **UVO - UV Overlays** in the addon list

## Usage

1. Enter **Edit Mode** on a mesh
2. Open the **UV Editor**
3. Make sure **Overlays** are enabled in the UV Editor header
4. Click the **UVO button** in the tool header (right side) to open the overlay panel
5. Enable any combination of ID, Intersect, Padding, and Stretch overlays

## Compatibility

Requires Blender 4.2 or newer.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE)
