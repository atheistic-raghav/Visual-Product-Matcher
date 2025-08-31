from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps
import requests
import io
from tensorflow.keras.applications import ResNet50  # Changed from MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input  # ResNet50 preprocessing
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PRODUCT_FOLDER'] = 'data/products'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Global variables for embeddings and model
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
    """Load product embeddings and initialize the ResNet50 feature extraction model"""
    global product_embeddings, product_filenames, product_names, product_categories, feature_model
    
    try:
        # Load product embeddings
        embeddings_path = 'data/product_embeddings.npz'
        print("🚀 Loading ResNet50 embeddings...")
        
        # Check if embeddings file exists
        if not os.path.exists(embeddings_path):
            print("⚠️ No embeddings found. Please run 'python feature_extractor.py' first!")
            print("⚠️ App will run but search functionality will be limited.")
            return
            
        data = np.load(embeddings_path)
        product_embeddings = data['embeddings']
        product_filenames = data['filenames']
        product_names = data['names']
        product_categories = data['categories']
        
        # Initialize ResNet50 model for feature extraction
        print("🧠 Initializing ResNet50 model...")
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # Changed to ResNet50
        feature_model = Model(inputs=base_model.input, outputs=base_model.output)
        
        print(f"✅ Loaded {len(product_embeddings)} ResNet50 embeddings successfully!")
        print(f"📊 Embedding dimensions: {product_embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {str(e)}")
        print("⚠️ App will run but search functionality will be limited.")
        product_embeddings = None

def enhanced_preprocess_image_for_search(img_path):
    """Enhanced preprocessing for better accuracy - matches feature_extractor.py"""
    try:
        # Load and convert to RGB
        img = Image.open(img_path).convert('RGB')
        
        # Smart resizing with high-quality resampling
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        
        # Enhance image quality for better feature extraction
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)  # Slight sharpness boost
        
        # Optional: Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.05)  # Subtle contrast boost
        
        # Preprocess for ResNet50
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)  # ResNet50-specific preprocessing
        
        # Extract features using ResNet50
        embedding = feature_model.predict(arr, verbose=0)[0]
        return embedding
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def find_similar_products_enhanced(query_embedding, top_k=12):  # CHANGED: Back to 12 from 24
    """Enhanced similarity search with better ranking"""
    if product_embeddings is None or query_embedding is None:
        return []
    
    try:
        # Reshape query embedding for cosine similarity calculation
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity between query and all product embeddings
        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        
        # Get indices of top-k most similar products
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results with enhanced information
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            
            results.append({
                'name': str(product_names[idx]),
                'category': str(product_categories[idx]),
                'image': f'/products/{product_filenames[idx]}',
                'similarity': similarity_score
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error finding similar products: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                return jsonify({
                    'success': True,
                    'image_path': f'uploads/{filename}',
                    'message': 'File uploaded successfully!'
                })
        
        # Handle URL upload
        elif 'image_url' in request.json:
            url = request.json['image_url']
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(io.BytesIO(response.content))
                filename = f"url_image_{hash(url) % 10000}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(filepath)
                return jsonify({
                    'success': True,
                    'image_path': f'uploads/{filename}',
                    'message': 'Image from URL saved successfully!'
                })
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error downloading image: {str(e)}'})
        
        return jsonify({'success': False, 'message': 'No valid file or URL provided'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Upload error: {str(e)}'})

@app.route('/search', methods=['POST'])
def search_similar():
    """Enhanced similarity search using ResNet50"""
    try:
        # Check if embeddings are loaded
        if product_embeddings is None:
            return jsonify({
                'success': False,
                'message': 'ResNet50 embeddings not loaded. Please run feature_extractor.py first.'
            })
        
        # Get the uploaded image path from request
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'success': False, 'message': 'No image path provided'})
        
        image_path = data['image_path']
        # Convert from URL path to file system path
        if image_path.startswith('uploads/'):
            full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
        else:
            full_image_path = os.path.join('static', image_path)
        
        # Check if file exists
        if not os.path.exists(full_image_path):
            return jsonify({'success': False, 'message': f'Image file not found: {image_path}'})
        
        # Extract features from uploaded image using enhanced preprocessing
        print(f"🔍 Processing image with ResNet50: {full_image_path}")
        query_embedding = enhanced_preprocess_image_for_search(full_image_path)
        if query_embedding is None:
            return jsonify({'success': False, 'message': 'Failed to process uploaded image'})
        
        # Find similar products using enhanced similarity search - CHANGED: Back to 12 results
        similar_products = find_similar_products_enhanced(query_embedding, top_k=12)
        
        if not similar_products:
            return jsonify({
                'success': True,
                'results': [],
                'message': 'No similar products found'
            })
        
        print(f"✅ Found {len(similar_products)} similar products with ResNet50")
        return jsonify({
            'success': True,
            'results': similar_products,
            'message': f'Found {len(similar_products)} similar products using ResNet50'
        })
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return jsonify({'success': False, 'message': f'Search error: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    model_status = "✅ Ready" if feature_model is not None else "❌ Not loaded"
    embeddings_status = "✅ Ready" if product_embeddings is not None else "❌ Not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'embeddings': embeddings_status,
        'products': len(product_embeddings) if product_embeddings is not None else 0
    })

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRODUCT_FOLDER'], exist_ok=True)
    
    # Load embeddings and model on startup
    print("🚀 Starting Visual Product Matcher with ResNet50...")
    load_embeddings_and_model()
    
    # Production vs Development configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌐 Server starting on port {port}")
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
