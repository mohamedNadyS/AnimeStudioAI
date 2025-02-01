from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import replicate
import requests
from moviepy.editor import *
from blender_script import generate_animation

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = "static\output"

APIS = {
    "REPLICATE_KEY": os.getenv('REPLICATE_API_KEY'),
    "ELEVENLABS_KEY": os.getenv('ELEVENLABS_API_KEY'),
    "BLENDER_MODEL": "https://github.com/OpenAnimationModels/Anime_Character_v2/raw/main/character.blend"
}

# Enhanced Story Generation System
def generate_story(prompt):
    response = replicate.run(
        "meta/llama-2-70b-chat",
        input={
            "prompt": f"Generate detailed anime screenplay in markdown format about: {prompt}",
            "system_prompt": "You are professional anime writer. Include: characters, dialogues, scenes, camera angles",
            "max_length": 4000
        }
    )
    return ''.join(response)

# Advanced Image Generation
def generate_scene_image(scene_desc, style):
    output = replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={
            "prompt": f"Anime {style} style, 4k detailed, {scene_desc}",
            "negative_prompt": "text, watermark, low quality",
            "num_outputs": 1
        }
    )
    return output[0]

# Professional Voice Generation
def generate_voice_over(text):
    headers = {"xi-api-key": APIS["ELEVENLABS_KEY"]}
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
        headers=headers,
        json={"text": text, "model_id": "eleven_monolingual_v1"}
    )
    return response.content

# Main Animation Workflow
@app.route('/generate', methods=['POST'])
def generate_animation():
    user_input = request.json
    session_id = str(uuid.uuid4())
    
    # Step 1: Generate Story
    screenplay = generate_story(user_input['prompt'])
    
    # Step 2: Generate Visual Assets
    scenes = screenplay.split('## Scene ')[1:5]  # Take first 4 scenes
    image_paths = []
    for idx, scene in enumerate(scenes):
        img_url = generate_scene_image(scene, user_input['style'])
        img_path = f"{app.config['UPLOAD_FOLDER']}/scene_{idx}.png"
        download_image(img_url, img_path)
        image_paths.append(img_path)
    
    # Step 3: Generate Voiceover
    audio_content = generate_voice_over(screenplay[:2000])
    audio_path = f"{app.config['UPLOAD_FOLDER']}/voice_{session_id}.mp3"
    with open(audio_path, 'wb') as f:
        f.write(audio_content)
    
    # Step 4: Create Animation
    video_path = generate_animation(image_paths, audio_path)  # Blender integration
    
    return jsonify({
        "video": video_path,
        "screenplay": screenplay,
        "assets": image_paths
    })

def download_image(url, path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

@app.route('/')
def index(): 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
