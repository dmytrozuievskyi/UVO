import bpy
import math


class UV_OT_ToggleOverlay(bpy.types.Operator):
    """Toggle mute on all UV ID overlays"""
    bl_idname = "uv.toggle_id_overlay"
    bl_label  = "UV ID Overlays"

    def execute(self, context):
        props = context.scene.uv_id_props
        props.is_muted = not props.is_muted
        return {'FINISHED'}


class UV_OT_RefreshOverlay(bpy.types.Operator):
    """Force recalculate overlay"""
    bl_idname = "uv.refresh_id_overlay"
    bl_label  = "Refresh UV ID Overlay"

    def execute(self, context):
        from . import draw
        draw.full_refresh(context)
        return {'FINISHED'}


class UV_OT_SampleStretchTexel(bpy.types.Operator):
    """Sample texel density from the selected UV island and fill the target field"""
    bl_idname = "uv.sample_stretch_texel"
    bl_label  = "Sample Texel Density"
    bl_description = (
        "Calculate the actual texel density of the selected UV island "
        "and write it into the Stretch target field"
    )

    @classmethod
    def poll(cls, context):
        return (
            context.mode == 'EDIT_MESH'
            and context.active_object is not None
        )

    def execute(self, context):
        # ----------------------------------------------------------------
        # Phase 1 stub — real Jacobian-based calculation comes in Phase 2+.
        # ----------------------------------------------------------------
        if not context.active_object:
            return {'CANCELLED'}
            
        obj_props = context.active_object.uv_id_props

        try:
            density_px_per_m = self._sample(context, obj_props)
        except Exception as exc:
            self.report({'WARNING'}, f"Sample failed: {exc}")
            return {'CANCELLED'}

        # Save internal density
        obj_props.stretch_internal_texel = density_px_per_m
        
        # Auto-switch unit for readability
        if density_px_per_m >= 1000.0:
            obj_props.stretch_texel_unit = 'PX_CM'
            obj_props.stretch_target_texel = density_px_per_m / 100.0
            display_val = density_px_per_m / 100.0
            unit_label = "px/cm"
        else:
            obj_props.stretch_texel_unit = 'PX_M'
            obj_props.stretch_target_texel = density_px_per_m
            display_val = density_px_per_m
            unit_label = "px/m"
            
        self.report({'INFO'}, f"Sampled: {display_val:.1f} {unit_label}")
        return {'FINISHED'}

    # ------------------------------------------------------------------
    def _sample(self, context, props):
        """
        Calculate texel density (px/m) for the selected UV islands.

        Formula:
            density [px/m] = sqrt( (tex_w * tex_h) * (uv_area / surface_area_3d) )

        - tex_w / tex_h  : texture dimensions from the global Texture Setup section
        - uv_area        : sum of triangle areas in UV [0,1] space
        - surface_area_3d: sum of triangle areas in 3D object space (metres)

        The maximum density across all selected islands is taken as the target
        (the highest-quality island drives the reference — per spec §4.5).

        Internal storage is always px/m. Unit dropdown only affects display.
        """
        import bmesh

        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            raise RuntimeError("No active mesh object")

        tex_w = int(props.tex_res_x)
        tex_h = int(props.tex_res_y)

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        uv_layer = bm.loops.layers.uv.verify()

        # Collect selected faces
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            raise RuntimeError("No faces selected")

        # --- island grouping (connected selected faces) ---
        islands = _find_uv_islands(selected_faces, uv_layer)

        densities = []
        for island_faces in islands:
            uv_area   = 0.0
            surf_area = 0.0

            for face in island_faces:
                loops = face.loops
                # Fan-triangulate the face
                l0 = loops[0]
                uv0  = l0[uv_layer].uv
                p0   = obj.matrix_world @ l0.vert.co

                for i in range(1, len(loops) - 1):
                    l1 = loops[i]
                    l2 = loops[i + 1]

                    uv1 = l1[uv_layer].uv
                    uv2 = l2[uv_layer].uv
                    p1  = obj.matrix_world @ l1.vert.co
                    p2  = obj.matrix_world @ l2.vert.co

                    # UV triangle area (cross product z-component, ×0.5)
                    eu = uv1 - uv0
                    ev = uv2 - uv0
                    uv_area += abs(eu.x * ev.y - eu.y * ev.x) * 0.5

                    # 3D triangle area
                    e1 = p1 - p0
                    e2 = p2 - p0
                    surf_area += e1.cross(e2).length * 0.5

            if surf_area < 1e-12 or uv_area < 1e-12:
                continue  # zero-area island — skip (per spec §4.4)

            density = math.sqrt((tex_w * tex_h) * (uv_area / surf_area))
            densities.append(density)

        if not densities:
            raise RuntimeError("All selected islands have zero area")

        return max(densities)  # spec: take maximum across selection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_uv_islands(faces, uv_layer):
    """
    Group faces into UV islands using flood-fill over shared UV vertices.
    Two faces belong to the same island if they share a UV loop coordinate.
    """
    face_set   = set(faces)
    visited    = set()
    islands    = []

    # Build a map: uv_coord_rounded → list of faces
    from collections import defaultdict
    uv_to_faces = defaultdict(list)
    for face in faces:
        for loop in face.loops:
            key = (round(loop[uv_layer].uv.x, 6), round(loop[uv_layer].uv.y, 6))
            uv_to_faces[key].append(face)

    for start in faces:
        if start in visited:
            continue
        island  = []
        queue   = [start]
        visited.add(start)
        while queue:
            current = queue.pop()
            island.append(current)
            for loop in current.loops:
                key = (round(loop[uv_layer].uv.x, 6), round(loop[uv_layer].uv.y, 6))
                for neighbour in uv_to_faces[key]:
                    if neighbour in face_set and neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
        islands.append(island)

    return islands


def register():
    bpy.utils.register_class(UV_OT_ToggleOverlay)
    bpy.utils.register_class(UV_OT_RefreshOverlay)
    bpy.utils.register_class(UV_OT_SampleStretchTexel)


def unregister():
    bpy.utils.unregister_class(UV_OT_SampleStretchTexel)
    bpy.utils.unregister_class(UV_OT_RefreshOverlay)
    bpy.utils.unregister_class(UV_OT_ToggleOverlay)