import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

# --- 1. CONFIGURE ABSOLUTE PATHS (FIX FILE NOT FOUND ERROR) ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BACKEND_DIR, '../frontend')


app = Flask(__name__, 
            template_folder=FRONTEND_DIR, 
            static_folder=FRONTEND_DIR,
            static_url_path='')
CORS(app)

# --- 2. LOAD WEIGHTS (MODIFIED TO USE ABSOLUTE PATHS) ---
models = {}

def load_weights(name, filename):
    filepath = os.path.join(BACKEND_DIR, filename)
    try:
        data = np.load(filepath)
        models[name] = {'W': data['W'], 'b': data['b']}
        print(f"Downloaded {name}: W shape {models[name]['W'].shape}")
    except Exception as e:
        print(f"Could not find {filename} at {filepath}. Error: {e}")

print("Starting system...")
load_weights('pixel', '../../models/model_function1.npz')
load_weights('sobel', '../../models/model_function2.npz')
load_weights('block', '../../models/model_function3.npz')

# --- 3. MATH & IMAGE PROCESSING FUNCTIONS (KEEP UNCHANGED) ---
def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def preprocess_common(image_file):
    # 1. Open image and resize to standard
    img = Image.open(image_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    # 2. Preprocess image
    top_left = img_array[0, 0]
    top_right = img_array[0, -1]
    bottom_left = img_array[-1, 0]
    bottom_right = img_array[-1, -1]
    
    # Calculate average of 4 corners
    avg_corners = (int(top_left) + int(top_right) + int(bottom_left) + int(bottom_right)) / 4
    
    if avg_corners > 127:
        # White background -> Invert to black background with white text (like MNIST)
        img_array = 255.0 - img_array

    # Convert to uint8
    img_uint8 = img_array.astype(np.uint8)

    _, img_bin = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Normalize to 0-1
    img_norm = img_bin / 255.0
    
    return img_norm


def process_pixel(img_norm):
    return img_norm.reshape(1, -1)

def process_sobel(img_norm):
    img_float = img_norm.astype(np.float32)
    sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = edges / (edges.max() + 1e-8)
    img_stacked = np.stack([img_norm, edges], axis=-1)
    return img_stacked.reshape(1, -1)

def process_block_avg(img_norm, block_size=2):
    H, W = img_norm.shape
    new_h, new_w = H // block_size, W // block_size
    valid_h, valid_w = new_h * block_size, new_w * block_size
    img_cropped = img_norm[:valid_h, :valid_w]
    reshaped = img_cropped.reshape(new_h, block_size, new_w, block_size)
    img_blocked = reshaped.mean(axis=(1, 3))
    return img_blocked.reshape(1, -1)

# --- 4. ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    model_type = request.form.get('model_type', 'pixel')
    
    if model_type not in models:
        return jsonify({'error': f"Model '{model_type}' has not been loaded. Please check the server log."}), 400

    file = request.files['file']
    
    try:
        img_norm = preprocess_common(file)
        
        if model_type == 'sobel':
            vector = process_sobel(img_norm)
        elif model_type == 'block':
            vector = process_block_avg(img_norm)
        else:
            vector = process_pixel(img_norm)
            
        W = models[model_type]['W']
        b = models[model_type]['b']
        
        if vector.shape[1] != W.shape[0]:
             return jsonify({'error': f"Shape Mismatch: Image {vector.shape}, Model {W.shape}"}), 500

        logits = np.dot(vector, W) + b
        probs = softmax(logits)[0]
        prediction = np.argmax(probs)
        
        return jsonify({
            'digit': int(prediction),
            'probabilities': probs.tolist(),
            'model_used': model_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
