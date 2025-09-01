from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import requests
import io
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import gc
import threading
import time

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
model_lock = threading.Lock()  # Thread safety

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/products/<filename>')
def serve_product_image(filename):
    try:
        return send_from_directory(app.config['PRODUCT_FOLDER'], filename)
    except Exception as e:
        print(f"❌ Error serving image {filename}: {e}")
        return "Image not found", 404

def load_embeddings_and_model():
    global product_embeddings, product_filenames, product_names, product_categories, feature_model
    try:
        print("🚀 Starting Visual Product Matcher with MobileNetV2...")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PRODUCT_FOLDER'], exist_ok=True)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        print(f"📊 Checking data directory: {data_dir}")
        
        if not os.path.exists(data_dir):
            print("❌ Data directory not found!")
            return
        
        embeddings_path = os.path.join(data_dir, 'product_embeddings.npz')
        print(f"🔍 Looking for embeddings at: {embeddings_path}")
        if not os.path.exists(embeddings_path):
            print("❌ Embeddings file not found!")
            print("🔄 Try running: python feature_extractor.py")
            return
        
        print("✅ Found embeddings file!")
        print("📥 Loading embeddings...")
        data = np.load(embeddings_path)
        product_embeddings = data['embeddings']
        product_filenames = data['filenames']
        product_names = data['names']
        product_categories = data['categories']
        print(f"✅ Loaded {len(product_embeddings)} embeddings successfully!")
        print(f"📊 Embedding shape: {product_embeddings.shape}")
        
        print("🧠 Initializing MobileNetV2 model (CPU-only)...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        feature_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("✅ MobileNetV2 model loaded!")

        gc.collect()
        print("🧹 Memory cleanup completed")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        product_embeddings = None

def optimized_preprocess_image(img_path):
    try:
        print(f"🖼️ Preprocessing image: {img_path}")
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("🔄 Converted image to RGB")
            if img.width < 64 or img.height < 64:
                print(f"⚠️ Image too small: {img_path}")
                return None
            img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = ImageEnhance.Sharpness(img).enhance(1.1)
            img = ImageEnhance.Contrast(img).enhance(1.05)
            img = ImageEnhance.Color(img).enhance(1.03)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            print("✅ Image preprocessing completed")
            return arr
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def fast_extract_embedding(preprocessed_img):
    global feature_model
    if preprocessed_img is None or feature_model is None:
        print("❌ Cannot extract embedding: missing data or model")
        return None
    try:
        print("🧠 Extracting features with MobileNetV2...")
        with model_lock:
            embedding = feature_model.predict(preprocessed_img, batch_size=1, verbose=0)[0]
        print(f"✅ Feature extraction completed, embedding shape: {embedding.shape}")
        gc.collect()
        return embedding
    except Exception as e:
        print(f"❌ Error extracting embedding: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def find_similar_products_fast(query_embedding, top_k=12):
    if product_embeddings is None or query_embedding is None:
        print("❌ Cannot find similar products: missing embeddings or query")
        return []
    try:
        print(f"🔍 Searching for {top_k} similar products...")
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            results.append({
                'name': str(product_names[idx]),
                'category': str(product_categories[idx]),
                'image': f'/products/{product_filenames[idx]}',
                'similarity': similarity_score
            })
        print(f"✅ Found {len(results)} similar products")
        return results
    except Exception as e:
        print(f"❌ Error finding similar products: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("📤 Processing upload request...")
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"💾 Saving uploaded file: {filename}")
                file.save(filepath)
                gc.collect()
                return jsonify({
                    'success': True,
                    'image_path': f'uploads/{filename}',
                    'message': 'File uploaded successfully!'
                })
        elif request.is_json and 'image_url' in request.json:
            url = request.json['image_url']
            print(f"🌐 Downloading image from URL: {url}")
            try:
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                filename = f"url_image_{abs(hash(url)) % 10000}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(filepath, 'JPEG', quality=85, optimize=True)
                print(f"💾 Saved URL image as: {filename}")
                gc.collect()
                return jsonify({
                    'success': True,
                    'image_path': f'uploads/{filename}',
                    'message': 'Image from URL saved successfully!'
                })
            except requests.exceptions.RequestException as e:
                print(f"❌ Error downloading image: {e}")
                return jsonify({'success': False, 'message': f'Error downloading image: {str(e)}'})
            except Exception as e:
                print(f"❌ Error processing URL image: {e}")
                return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})
        return jsonify({'success': False, 'message': 'No valid file or URL provided'})
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'success': False, 'message': f'Upload error: {str(e)}'})

@app.route('/search', methods=['POST'])
def search_similar():
    start_time = time.time()
    try:
        print("🔍 Starting similarity search...")
        if product_embeddings is None or feature_model is None:
            return jsonify({
                'success': False,
                'message': 'System not ready. Please try later.'
            })
        data = request.get_json()
        image_path = data.get('image_path')
        if not image_path:
            return jsonify({'success': False, 'message': 'No image path provided'})
        if image_path.startswith('uploads/'):
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
        else:
            full_path = os.path.join('static', image_path)
        if not os.path.exists(full_path):
            return jsonify({'success': False, 'message': 'Image file not found'})
        preprocessed = optimized_preprocess_image(full_path)
        if preprocessed is None:
            return jsonify({'success': False, 'message': 'Failed to preprocess image'})
        query_embedding = fast_extract_embedding(preprocessed)
        if query_embedding is None:
            return jsonify({'success': False, 'message': 'Failed to extract features'})
        results = find_similar_products_fast(query_embedding)
        total_time = time.time() - start_time
        return jsonify({'success': True, 'results': results, 'search_time': f'{total_time:.2f}s'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Search error: {e}'})

@app.route('/health')
def health_check():
    model_ready = feature_model is not None
    embeddings_ready = product_embeddings is not None
    memory_info = "not available"
    return jsonify({
        'status': 'healthy' if embeddings_ready and model_ready else 'starting',
        'model': 'ready' if model_ready else 'not loaded',
        'embeddings': 'ready' if embeddings_ready else 'not loaded',
        'products': len(product_embeddings) if embeddings_ready else 0,
        'memory_usage': memory_info,
        'system': 'MobileNetV2 Visual Product Matcher',
        'version': '2.0'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

print("🔄 Initializing Visual Product Matcher...")
load_embeddings_and_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f"🌐 Starting development server on port {port}")
    print(f"🐛 Debug mode: {'ON' if debug else 'OFF'}")
    app.run(debug=debug, host='0.0.0.0', port=port)
