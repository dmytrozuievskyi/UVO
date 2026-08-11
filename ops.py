import bpy
import math


class UV_OT_ToggleOverlay(bpy.types.Operator):
    """Toggle mute on all UV overlays"""
    bl_idname = "uv.toggle_overlay"
    bl_label  = "UV Overlays"

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
        if not context.active_object:
            return {'CANCELLED'}
            
        obj_props = context.active_object.uv_id_props

        try:
            density_px_per_m = self._sample(context)
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


    def _sample(self, context):
        """
        Calculate texel density (px/m) for the selected UV islands across all edit objects.

        Formula:
            density [px/m] = sqrt( (tex_w * tex_h) * (uv_area / surface_area_3d) )

        - tex_w / tex_h  : texture dimensions from the object's Texture Setup section
        - uv_area        : sum of triangle areas in UV [0,1] space
        - surface_area_3d: sum of triangle areas in 3D object space (metres)

        The maximum density across all sampled islands across all objects is returned.
        """
        import bmesh

        edit_objs = [o for o in context.scene.objects if o.type == 'MESH' and o.mode == 'EDIT']
        if not edit_objs:
            raise RuntimeError("No edit mesh objects found")
            
        sync_on = context.scene.tool_settings.use_uv_select_sync
        
        has_any_selection = False
        obj_data = {} # {obj: (bm, uv_layer, sel_faces)}
        
        # Pass 1: Find selections respecting UV Sync
        for obj in edit_objs:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.verify()
            
            if sync_on:
                sel_faces = [f for f in bm.faces if f.select]
            else:
                sel_faces = []
                for f in bm.faces:
                    face_selected = False
                    for l in f.loops:
                        if hasattr(l, "uv_select_vert"):
                            if getattr(l, "uv_select_vert"):
                                face_selected = True
                                break
                        else:
                            if getattr(l[uv_layer], "select", False):
                                face_selected = True
                                break
                    if face_selected:
                        sel_faces.append(f)
                
            if sel_faces:
                has_any_selection = True
                
            obj_data[obj] = (bm, uv_layer, sel_faces)
            
        # Pass 2: Fallback if nothing selected anywhere
        if not has_any_selection:
            for obj in edit_objs:
                bm, uv_layer, _ = obj_data[obj]
                obj_data[obj] = (bm, uv_layer, list(bm.faces))

        max_density = 0.0
        found_any = False

        for obj in edit_objs:
            bm, uv_layer, sel_faces = obj_data[obj]
            if not sel_faces:
                continue
                
            props = obj.uv_id_props if hasattr(obj, 'uv_id_props') else context.active_object.uv_id_props
            tex_w = int(props.tex_res_x)
            tex_h = int(props.tex_res_y)

            islands = _find_uv_islands(sel_faces, uv_layer)

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

                if surf_area > 1e-12 and uv_area > 1e-12:
                    density = math.sqrt((tex_w * tex_h) * (uv_area / surf_area))
                    max_density = max(max_density, density)
                    found_any = True

        if not found_any:
            raise RuntimeError("All sampled islands have zero area")

        return max_density



def _find_uv_islands(faces, uv_layer):
    """Group faces into UV islands using flood-fill over shared UV vertices."""
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


class VIEW3D_OT_toggle_uv_seams_overlay(bpy.types.Operator):
    bl_idname = "view3d.toggle_uv_seams_overlay"
    bl_label = "Toggle UV Seams Overlay"
    bl_description = "Toggle dynamic UV seam visualization in this viewport"
    
    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' 
                and context.space_data is not None 
                and context.space_data.type == 'VIEW_3D')
    
    def execute(self, context):
        from . import draw_3d
        draw_3d.toggle_viewport(context)
        return {'FINISHED'}


class VIEW3D_OT_toggle_uvo_3d_mute(bpy.types.Operator):
    """Toggle the global mute state for UVO 3D Overlays"""
    bl_idname = "view3d.toggle_uvo_3d_mute"
    bl_label = "Toggle UVO 3D Mute"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.uv_3d_seam_props
        props.is_muted = not props.is_muted
        return {'FINISHED'}


def register():
    bpy.utils.register_class(UV_OT_ToggleOverlay)
    bpy.utils.register_class(UV_OT_RefreshOverlay)
    bpy.utils.register_class(UV_OT_SampleStretchTexel)
    bpy.utils.register_class(VIEW3D_OT_toggle_uv_seams_overlay)
    bpy.utils.register_class(VIEW3D_OT_toggle_uvo_3d_mute)


def unregister():
    bpy.utils.unregister_class(VIEW3D_OT_toggle_uvo_3d_mute)
    bpy.utils.unregister_class(VIEW3D_OT_toggle_uv_seams_overlay)
    bpy.utils.unregister_class(UV_OT_SampleStretchTexel)
    bpy.utils.unregister_class(UV_OT_RefreshOverlay)
    bpy.utils.unregister_class(UV_OT_ToggleOverlay)