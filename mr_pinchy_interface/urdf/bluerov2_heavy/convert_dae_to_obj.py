import bpy
import json
import re
import sys
import os

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if len(argv) != 2:
    raise SystemExit("Usage: blender -b -P convert_dae_to_obj.py -- input.dae output.obj")

in_path = os.path.abspath(argv[0])
out_path = os.path.abspath(argv[1])
out_dir = os.path.dirname(out_path)
out_basename = os.path.splitext(os.path.basename(out_path))[0]
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.collada_import(filepath=in_path)


def sanitize_name(name):
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return safe or "mat"


used_names = set()
for material in bpy.data.materials:
    candidate = sanitize_name(material.name)
    base = candidate
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    material.name = candidate
    used_names.add(candidate)


def export_obj(filepath, use_selection):
    if bpy.app.version >= (3, 3, 0):
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_materials=True,
            export_selected_objects=use_selection,
            path_mode='RELATIVE',
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=use_selection,
            use_materials=True,
            path_mode='RELATIVE',
        )


mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not mesh_objects:
    raise SystemExit("No mesh objects imported from DAE.")

bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]

if len(mesh_objects) > 1:
    bpy.ops.object.join()

if len(bpy.context.view_layer.objects.active.data.materials) > 1:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='MATERIAL')
    bpy.ops.object.mode_set(mode='OBJECT')


def part_diffuse(material):
    if material is None:
        return [0.5, 0.5, 0.5]
    color = list(material.diffuse_color)
    return [float(color[0]), float(color[1]), float(color[2])]


parts = []
seen_filenames = set()
for obj in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
    if not obj.data.polygons:
        continue
    if obj.data.materials:
        material = obj.data.materials[0]
    else:
        material = None
    material_name = material.name if material is not None else f"{out_basename}_part"
    safe_part_name = sanitize_name(material_name)
    part_filename = f"{out_basename}_{safe_part_name}.obj"
    suffix = 1
    while part_filename in seen_filenames:
        part_filename = f"{out_basename}_{safe_part_name}_{suffix}.obj"
        suffix += 1
    seen_filenames.add(part_filename)
    part_path = os.path.join(out_dir, part_filename)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    export_obj(part_path, use_selection=True)

    parts.append({
        "file": part_filename,
        "material": safe_part_name,
        "kd": part_diffuse(material),
    })

bpy.ops.object.select_all(action='SELECT')
export_obj(out_path, use_selection=False)

manifest = {
    "source": os.path.basename(out_path),
    "parts": parts,
}
manifest_path = os.path.join(out_dir, f"{out_basename}.parts.json")
with open(manifest_path, "w") as fp:
    json.dump(manifest, fp, indent=2)

print(f"Exported: {out_path}")
print(f"Manifest: {manifest_path}")
print(f"Parts: {len(parts)}")
