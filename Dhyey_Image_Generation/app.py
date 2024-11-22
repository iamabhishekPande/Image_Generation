from flask import Flask, request, send_file, jsonify
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
from authtoken import auth_token
import os

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Preloading the model (adjust according to actual deployment needs)
model_cache = {}
print("Model cache loaded")

def load_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name], None

    model_path = os.path.join(f'/app/models/{model_name}')
    print(f"Loading model from: {model_path}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
        pipe.to(device)
        model_cache[model_name] = pipe
        return pipe, None
    except Exception as e:
        return None, str(e)


@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    model_name = data.get('model_name')
    text_prompt = data.get('prompt')

    if not model_name or not text_prompt:
        return jsonify({"error": "Model name and prompt are required"}), 400

    pipe, error = load_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    try:
        with autocast(device):
            output = pipe(text_prompt, guidance_scale=8.0)
            image = output.images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/hello", methods=["GET"])
# def say_hello():
#     return jsonify({"message": "Hello from Flask"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
