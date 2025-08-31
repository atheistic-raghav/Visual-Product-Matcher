import os

# Limit TensorFlow threading for lower memory usage
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import io
import traceback
import numpy as np
import requests
import gc

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
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
    """Serve product images from data/products folder"""
    return send_from_directory(app.config['PRODUCT_FOLDER'], filename)


def load_embeddings_and_model():
    global product_embeddings, product_filenames, product_names, product_categories, feature_model
    try:
        print("🚀 Starting Visual Matcher with ResNet50...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔍 Current directory: {current_dir}")

        try:
            contents = os.listdir(current_dir)
            print(f"📂 Root directory contents: {contents}")
        except Exception as e:
            print(f"❌ Unable to list directory: {e}")

        data_dir = os.path.join(current_dir, 'data')
        print(f"📊 Checking data directory: {data_dir}")

        if not os.path.exists(data_dir):
            print("❌ Data directory does not exist.")
            return

        data_contents = os.listdir(data_dir)
        print(f"📂 Data directory contents: {data_contents}")

        embeddings_path = os.path.join(data_dir, 'product_embeddings.npz')
        print(f"🔍 Looking for embeddings at: {embeddings_path}")

        if not os.path.exists(embeddings_path):
            print("❌ Embeddings file does not exist. Please run the feature extractor.")
            return

        data = np.load(embeddings_path)
        product_embeddings = data['embeddings']
        product_filenames = data['filenames']
        product_names = data['names']
        product_categories = data['categories']

        print(f"✅ Loaded embeddings, count: {len(product_embeddings)}")
        print(f"Embedding shape: {product_embeddings.shape}")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(product_embeddings, axis=1, keepdims=True)
        product_embeddings[:] = product_embeddings / np.clip(norms, 1e-12, None)

        print("🧠 Initializing ResNet50 model...")
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("✅ ResNet50 model loaded.")

        gc.collect()

    except Exception as e:
        print(f"❌ Exception during loading embeddings/model: {e}")
        traceback.print_exc()
        product_embeddings = None


def enhanced_preprocess_image_for_search(img_path):
    """Preprocess and generate embedding for a given image."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.05)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        embedding = feature_model.predict(arr, verbose=0)[0]
        embedding /= np.linalg.norm(embedding) + 1e-12
        return embedding
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        traceback.print_exc()
        return None


def find_similar_products_enhanced(query_embedding, top_k=12):
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
            sim = max(0.0, min(sim, 1.0))
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
            url = data.get('image_url') if data else None
            if url:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                filename = f"url_{abs(hash(url)) % 10000}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(filepath)
                gc.collect()
                return jsonify(success=True, image_path=f'uploads/{filename}', message='Image saved from URL.')

        return jsonify(success=False, message='No valid file or image URL provided'), 400
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return jsonify(success=False, message=str(e)), 500


@app.route('/search', methods=['POST'])
def search_similar():
    try:
        if product_embeddings is None:
            return jsonify(success=False, message='ResNet50 embeddings not loaded. Please run the feature extractor.')

        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify(success=False, message='No image path provided')

        image_path = data['image_path']
        if image_path.startswith('uploads/'):
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
        else:
            full_path = os.path.join('static', image_path)

        if not os.path.exists(full_path):
            return jsonify(success=False, message=f'Image file not found: {image_path}')

        embedding = enhanced_preprocess_image_for_search(full_path)
        if embedding is None:
            return jsonify(success=False, message='Failed to process image')

        results = find_similar_products_enhanced(embedding, top_k=12)
        if not results:
            return jsonify(success=True, results=[], message='No similar products found')

        return jsonify(success=True, results=results, message=f'Found {len(results)} similar products')
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


# Call to load model and embeddings when module loads (important for Gunicorn)
load_embeddings_and_model()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRODUCT_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f'Launching server on port {port}, debug={debug}')
    app.run(host='0.0.0.0', port=port, debug=debug)
