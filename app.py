# app.py
from flask import Flask, render_template, request, jsonify
import os
import uuid
import replicate
import requests
from moviepy.editor import *
from lottie import Animation

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/output'

class AnimationGenerator:
    def __init__(self):
        self.animation_engine = Animation(api_key=os.getenv('LOTTIE_API_KEY'))
        
    def generate_story(self, prompt):
        response = replicate.run(
            "meta/llama-2-70b-chat",
            input={"prompt": f"Write anime screenplay: {prompt}"}
        )
        return ''.join(response)
    
    def generate_scene(self, scene_desc, style):
        return replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={"prompt": f"Anime {style} style, {scene_desc}"}
        )[0]
    
    def create_animation(self, images, audio_path, output_name):
        # Create Lottie animation
        animation_data = self.animation_engine.create(
            assets=images,
            animation_type="slide_show",
            duration=5
        )
        
        # Render to video
        animation_clip = self.animation_engine.render(animation_data['id'])
        audio_clip = AudioFileClip(audio_path)
        
        final_clip = animation_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_name, fps=24)
        return output_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    session_id = str(uuid.uuid4())
    
    # Initialize components
    generator = AnimationGenerator()
    
    # Generate story
    screenplay = generator.generate_story(data['prompt'])
    
    # Generate scenes
    scenes = screenplay.split("\n\n")[:3]
    images = []
    for idx, scene in enumerate(scenes):
        img_url = generator.generate_scene(scene, data['style'])
        img_path = f"{app.config['UPLOAD_FOLDER']}/scene_{idx}.png"
        download_image(img_url, img_path)
        images.append(img_path)
    
    # Generate audio
    audio_content = generate_voice(screenplay[:2000])
    audio_path = f"{app.config['UPLOAD_FOLDER']}/voice_{session_id}.mp3"
    with open(audio_path, 'wb') as f:
        f.write(audio_content)
    
    # Create animation
    video_path = generator.create_animation(images, audio_path, "final_animation.mp4")
    
    return jsonify({
        "video": video_path,
        "screenplay": screenplay
    })

def download_image(url, path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

def generate_voice(text):
    headers = {"xi-api-key": os.getenv('ELEVENLABS_API_KEY')}
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
        headers=headers,
        json={"text": text}
    )
    return response.content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
