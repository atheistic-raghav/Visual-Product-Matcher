from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps
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

# Global variables
product_embeddings = None
product_filenames = None
product_names = None
product_categories = None
feature_model = None
model_lock = threading.Lock()  # Thread safety for model access

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/products/<filename>')
def serve_product_image(filename):
    """Serve product images from data/products folder"""
    try:
        return send_from_directory(app.config['PRODUCT_FOLDER'], filename)
    except Exception as e:
        print(f"❌ Error serving image {filename}: {e}")
        return "Image not found", 404

def load_embeddings_and_model():
    """Load product embeddings and initialize MobileNetV2 model"""
    global product_embeddings, product_filenames, product_names, product_categories, feature_model
    
    try:
        print("🚀 Starting Visual Product Matcher with MobileNetV2...")
        
        # Create required directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PRODUCT_FOLDER'], exist_ok=True)
        
        # Get current directory and check for data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        print(f"📊 Checking data directory: {data_dir}")
        
        if not os.path.exists(data_dir):
            print("❌ Data directory not found!")
            return
        
        # Look for embeddings file
        embeddings_path = os.path.join(data_dir, 'product_embeddings.npz')
        print(f"🔍 Looking for embeddings at: {embeddings_path}")
        
        if not os.path.exists(embeddings_path):
            print("❌ Embeddings file not found!")
            print("🔄 Try running: python feature_extractor.py")
            return
        
        print("✅ Found embeddings file!")
        # Load embeddings
        print("📥 Loading embeddings...")
        data = np.load(embeddings_path)
        product_embeddings = data['embeddings']
        product_filenames = data['filenames']
        product_names = data['names']
        product_categories = data['categories']
        print(f"✅ Loaded {len(product_embeddings)} embeddings successfully!")
        print(f"📊 Embedding shape: {product_embeddings.shape}")
        
        # Initialize MobileNetV2 model (CPU only)
        print("🧠 Initializing MobileNetV2 model...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        feature_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("✅ MobileNetV2 model loaded!")
        
        # Final status
        if product_embeddings is not None:
            print(f"🎉 SUCCESS! System ready with {len(product_embeddings)} products")
        else:
            print("⚠️ No embeddings available - search will be limited")
        
        # Force garbage collection to free memory
        gc.collect()
        print("🧹 Memory cleanup completed")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        product_embeddings = None

def optimized_preprocess_image(img_path):
    """Optimized preprocessing for MobileNetV2 - matches feature_extractor.py"""
    try:
        print(f"🖼️ Preprocessing image: {img_path}")
        
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("🔄 Converted image to RGB")
            
            # Skip extremely small images
            if img.width < 64 or img.height < 64:
                print(f"⚠️ Image too small: {img_path}")
                return None
            
            # Resize with aspect ratio preservation (same as feature_extractor.py)
            img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            
            # Enhanced preprocessing pipeline (matching feature_extractor.py)
            from PIL import ImageFilter
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = ImageEnhance.Sharpness(img).enhance(1.1)
            img = ImageEnhance.Contrast(img).enhance(1.05)
            img = ImageEnhance.Color(img).enhance(1.03)
            
            # Convert to array and preprocess for MobileNetV2
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            
            print("✅ Image preprocessing completed")
            return arr
            
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def fast_extract_embedding(preprocessed_img):
    """Fast embedding extraction with memory management and thread safety"""
    global feature_model
    
    if preprocessed_img is None or feature_model is None:
        print("❌ Cannot extract embedding: missing data or model")
        return None
    
    try:
        print("🧠 Extracting features with MobileNetV2...")
        
        with model_lock:  # Thread-safe model access
            # Run prediction
            embedding = feature_model.predict(preprocessed_img, batch_size=1, verbose=0)[0]
        
        print(f"✅ Feature extraction completed, embedding shape: {embedding.shape}")
        
        # Force garbage collection after prediction
        gc.collect()
        return embedding
        
    except Exception as e:
        print(f"❌ Error extracting embedding: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def find_similar_products_fast(query_embedding, top_k=12):
    """Lightning-fast similarity search with cosine similarity"""
    if product_embeddings is None or query_embedding is None:
        print("❌ Cannot find similar products: missing embeddings or query")
        return []
    
    try:
        print(f"🔍 Searching for {top_k} similar products...")
        
        # Reshape query embedding for cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        
        # Fast cosine similarity computation
        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        
        # Get top results efficiently
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
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with support for both file upload and URL input"""
    try:
        print("📤 Processing upload request...")
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"💾 Saving uploaded file: {filename}")
                file.save(filepath)
                
                # Force garbage collection
                gc.collect()
                
                return jsonify({
                    'success': True,
                    'image_path': f'uploads/{filename}',
                    'message': 'File uploaded successfully!'
                })
        
        # Handle URL input
        elif request.is_json and 'image_url' in request.json:
            url = request.json['image_url']
            print(f"🌐 Downloading image from URL: {url}")
            
            try:
                # Download with timeout
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                
                # Process downloaded image
                img = Image.open(io.BytesIO(response.content))
                filename = f"url_image_{abs(hash(url)) % 10000}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Convert to RGB and save
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(filepath, 'JPEG', quality=85, optimize=True)
                print(f"💾 Saved URL image as: {filename}")
                
                # Force garbage collection
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
    """Similarity search with comprehensive error handling and performance monitoring"""
    start_time = time.time()
    
    try:
        print("🔍 Starting similarity search...")
        
        # Check if system is ready
        if product_embeddings is None or feature_model is None:
            return jsonify({
                'success': False,
                'message': 'System not ready. Embeddings or model not loaded. Please wait and try again.'
            })
        
        # Get image path from request
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'success': False, 'message': 'No image path provided'})
        
        image_path = data['image_path']
        print(f"📸 Processing search for image: {image_path}")
        
        # Build full path to uploaded image
        if image_path.startswith('uploads/'):
            full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
        else:
            full_image_path = os.path.join('static', image_path)
        
        # Check if file exists
        if not os.path.exists(full_image_path):
            print(f"❌ Image file not found: {full_image_path}")
            return jsonify({'success': False, 'message': f'Image file not found: {image_path}'})
        
        print(f"✅ Image file found: {full_image_path}")
        
        # Step 1: Preprocess image
        preprocess_start = time.time()
        preprocessed = optimized_preprocess_image(full_image_path)
        if preprocessed is None:
            return jsonify({'success': False, 'message': 'Failed to preprocess image'})
        preprocess_time = time.time() - preprocess_start
        print(f"⏱️ Preprocessing took: {preprocess_time:.2f}s")
        
        # Step 2: Extract embedding
        embedding_start = time.time()
        query_embedding = fast_extract_embedding(preprocessed)
        if query_embedding is None:
            return jsonify({'success': False, 'message': 'Failed to extract features from image'})
        embedding_time = time.time() - embedding_start
        print(f"⏱️ Feature extraction took: {embedding_time:.2f}s")
        
        # Step 3: Find similar products
        search_start = time.time()
        similar_products = find_similar_products_fast(query_embedding, top_k=12)
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        print(f"⏱️ Similarity search took: {search_time:.2f}s")
        print(f"⏱️ Total search time: {total_time:.2f}s")
        
        if not similar_products:
            return jsonify({
                'success': True,
                'results': [],
                'message': 'No similar products found',
                'search_time': f"{total_time:.2f}s"
            })
        
        print(f"✅ Found {len(similar_products)} similar products")
        print("🔝 Top 3 matches:")
        for i, product in enumerate(similar_products[:3]):
            print(f"  {i+1}. {product['name']} ({product['similarity']:.3f})")
        
        # Force garbage collection before returning
        gc.collect()
        
        return jsonify({
            'success': True,
            'results': similar_products,
            'message': f'Found {len(similar_products)} similar products using MobileNetV2',
            'search_time': f"{total_time:.2f}s",
            'breakdown': {
                'preprocessing': f"{preprocess_time:.2f}s",
                'feature_extraction': f"{embedding_time:.2f}s",
                'similarity_search': f"{search_time:.2f}s"
            }
        })
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Search error after {total_time:.2f}s: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Force garbage collection on error
        gc.collect()
        
        return jsonify({
            'success': False,
            'message': f'Search failed: {str(e)}',
            'search_time': f"{total_time:.2f}s"
        })

@app.route('/health')
def health_check():
    """Enhanced health check endpoint with system status (no psutil dependency)"""
    model_status = "✅ Ready" if feature_model is not None else "❌ Not loaded"
    embeddings_status = "✅ Ready" if product_embeddings is not None else "❌ Not loaded"
    
    # Memory info without psutil dependency
    memory_info = "not available"
    
    return jsonify({
        'status': 'healthy' if product_embeddings is not None else 'loading',
        'model': model_status,
        'embeddings': embeddings_status,
        'products': len(product_embeddings) if product_embeddings is not None else 0,
        'memory_usage': memory_info,
        'system': 'MobileNetV2 Visual Product Matcher',
        'version': '2.0'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'success': False, 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

# Load embeddings when module is imported (Gunicorn compatible!)
print("🔄 Initializing Visual Product Matcher...")
load_embeddings_and_model()

if __name__ == '__main__':
    # This only runs in development mode with `python app.py`
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌐 Starting development server on port {port}")
    print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
