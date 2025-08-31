import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # less verbose logs


import io
import traceback
import numpy as np
import requests
import gc

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PRODUCT_FOLDER'] = 'data/products'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Globals
product_embeddings = None
product_filenames = None
product_names = None
product_categories = None
feature_model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/products/<path:filename>')
def serve_product_image(filename):
    return send_from_directory(app.config['PRODUCT_FOLDER'], filename)


def load_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)


def load_embeddings_and_model():
    global product_embeddings, product_filenames, product_names, product_categories, feature_model
    try:
        print("🚀 Loading embeddings and MobileNetV2 model")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        embeddings_path = os.path.join(data_dir, 'product_embeddings.npz')

        if not os.path.exists(embeddings_path):
            print("❌ Embeddings file not found. Run feature extractor first.")
            return

        data = np.load(embeddings_path)
        product_embeddings = data['embeddings']
        product_filenames = data['filenames']
        product_names = data['names']
        product_categories = data['categories']

        print(f"✅ Loaded {len(product_embeddings)} embeddings")
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(product_embeddings, axis=1, keepdims=True)
        product_embeddings[:] = product_embeddings / np.clip(norms, 1e-12, None)

        feature_model = load_model()
        print("✅ MobileNetV2 model loaded")
        gc.collect()

    except Exception as e:
        print(f"❌ Error during loading embeddings/model: {e}")
        traceback.print_exc()
        product_embeddings = None


def enhanced_preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.05)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        return arr
    except Exception as e:
        print(f"❌ Error in image preprocessing: {e}")
        traceback.print_exc()
        return None


def find_similar_products(query_embedding, top_k=12):
    if product_embeddings is None or query_embedding is None:
        return []
    try:
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        top_idxs = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_idxs:
            sim = similarities[idx]
            sim = max(0.0, min(sim, 1.0))  # Clamp to [0,1]
            results.append({
                "name": str(product_names[idx]),
                "category": str(product_categories[idx]),
                "image": f"/products/{product_filenames[idx]}",
                "similarity": round(sim * 100, 1)
            })
        return results
    except Exception as e:
        print(f"❌ Error finding similar products: {e}")
        traceback.print_exc()
        return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                gc.collect()
                return jsonify(success=True, image_path=f'uploads/{filename}', message='File uploaded successfully.')

        elif request.is_json:
            data = request.get_json()
            if data and "image_url" in data:
                url = data["image_url"]
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                filename = f"url_{abs(hash(url)) % 10000}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(filepath)
                gc.collect()
                return jsonify(success=True, image_path=f'uploads/{filename}', message='Image saved from URL.')

        return jsonify(success=False, message='No valid file or image URL provided.'), 400
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return jsonify(success=False, message=str(e)), 500


@app.route('/search', methods=['POST'])
def search():
    try:
        if product_embeddings is None:
            return jsonify(success=False, message='Embeddings not loaded. Run feature extractor first.')

        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify(success=False, message='No image path provided.')

        image_path = data['image_path']
        if image_path.startswith('uploads/'):
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
        else:
            full_path = os.path.join('static', image_path)

        if not os.path.exists(full_path):
            return jsonify(success=False, message=f'Image file not found: {image_path}')

        img_array = enhanced_preprocess_image(full_path)
        if img_array is None:
            return jsonify(success=False, message='Failed to preprocess image.')

        embedding = feature_model.predict(img_array, verbose=0)[0]
        embedding /= np.linalg.norm(embedding) + 1e-12

        results = find_similar_products(embedding, top_k=12)
        if not results:
            return jsonify(success=True, results=[], message='No similar products found.')

        return jsonify(success=True, results=results, message=f'Found {len(results)} similar products.')
    except Exception as e:
        print(f"❌ Search error: {e}")
        traceback.print_exc()
        return jsonify(success=False, message=str(e)), 500


@app.route('/health')
def health():
    return jsonify(
        status='healthy',
        model='loaded' if feature_model else 'not loaded',
        embeddings='loaded' if product_embeddings is not None else 'not loaded',
        total_products=len(product_embeddings) if product_embeddings is not None else 0
    )


# Load embeddings and model immediately on import (for Gunicorn)
load_embeddings_and_model()


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRODUCT_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f'Launching server on port {port}, debug={debug}')
    app.run(host='0.0.0.0', port=port, debug=debug)
