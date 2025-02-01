import bpy
import subprocess
import json

def generate_animation(image_paths, audio_path):
    # Load base animation model
    bpy.ops.wm.open_mainfile(filepath="base_animation.blend")
    
    # Set up scenes
    for idx, img_path in enumerate(image_paths):
        create_scene(img_path, duration=10, scene_number=idx)
    
    # Add audio track
    add_audio(audio_path)
    
    # Render animation
    output_path = "static/output/animation.mp4"
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)
    
    return output_path

def create_scene(image_path, duration, scene_number):
    # Create new scene
    bpy.ops.scene.new(type='EMPTY')
    scene = bpy.context.scene
    
    # Set up camera and lighting
    bpy.ops.object.camera_add()
    bpy.ops.object.light_add(type='AREA')
    
    # Add background image
    add_background_image(image_path)
    
    # Configure scene duration
    scene.frame_start = scene_number * 250
    scene.frame_end = (scene_number + 1) * 250
    
def add_background_image(image_path):
    # Create image plane
    bpy.ops.mesh.primitive_plane_add(size=10)
    plane = bpy.context.object
    mat = bpy.data.materials.new(name="BG_Material")
    plane.data.materials.append(mat)
    
    # Assign texture
    tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(image_path)
    mat.node_tree.links.new(
        mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'],
        tex.outputs['Color']
    )

def add_audio(audio_path):
    # Import audio file
    bpy.ops.sequencer.sound_strip_add(
        filepath=audio_path,
        frame_start=1
    )
