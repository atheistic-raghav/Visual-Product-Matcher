from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps
import requests
import io
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

# Global variables
product_embeddings = None
product_filenames = None
product_names = None
product_categories = None
feature_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/products/<path:filename>')  # ✅ CRITICAL FIX: Added missing <path:filename> parameter
def serve_product_image(filename):
    """Serve product images from data/products folder"""
    return send_from_directory(app.config['PRODUCT_FOLDER'], filename)

def load_embeddings_and_model():
    """Load product embeddings and initialize ResNet50 model"""
    global product_embeddings, product_filenames, product_names, product_categories, feature_model
    
    try:
        print("🚀 Starting Visual Product Matcher with ResNet50...")
        
        # STEP 1: Get current directory and list contents
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔍 Current directory: {current_dir}")
        
        # List all directories and files for debugging
        try:
            contents = os.listdir(current_dir)
            print(f"📂 Root directory contents: {contents}")
        except Exception as e:
            print(f"❌ Cannot list directory: {e}")
        
        # STEP 2: Check data directory
        data_dir = os.path.join(current_dir, 'data')
        print(f"📊 Checking data directory: {data_dir}")
        
        if os.path.exists(data_dir):
            data_contents = os.listdir(data_dir)
            print(f"📊 Data directory contents: {data_contents}")
            
            # STEP 3: Look for embeddings file
            embeddings_path = os.path.join(data_dir, 'product_embeddings.npz')
            print(f"🔍 Looking for embeddings at: {embeddings_path}")
            
            if os.path.exists(embeddings_path):
                print(f"✅ Found embeddings file!")
                
                # STEP 4: Load embeddings
                print("📥 Loading embeddings...")
                data = np.load(embeddings_path)
                product_embeddings = data['embeddings']
                product_filenames = data['filenames']
                product_names = data['names'] 
                product_categories = data['categories']
                
                print(f"✅ Loaded {len(product_embeddings)} embeddings successfully!")
                print(f"📊 Embedding shape: {product_embeddings.shape}")
                
            else:
                print("❌ Embeddings file not found!")
                print("🔄 Try running: python feature_extractor.py")
                return
                
        else:
            print("❌ Data directory not found!")
            return
            
        # STEP 5: Initialize ResNet50 model
        print("🧠 Initializing ResNet50 model...")
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("✅ ResNet50 model loaded successfully!")
        
        # STEP 6: Final status
        if product_embeddings is not None:
            print(f"🎉 SUCCESS! System ready with {len(product_embeddings)} products")
        else:
            print("⚠️ No embeddings available - search will be limited")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        product_embeddings = None

def enhanced_preprocess_image_for_search(img_path):
    """Enhanced preprocessing for search - matches feature_extractor.py"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        
        # Enhanced processing
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        contrast_enhancer = ImageEnhance.Contrast(img)  
        img = contrast_enhancer.enhance(1.05)
        
        # Convert to array and preprocess for ResNet50
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        
        # Extract embedding
        embedding = feature_model.predict(arr, verbose=0)[0]
        return embedding
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def find_similar_products_enhanced(query_embedding, top_k=12):
    """Enhanced similarity search"""
    if product_embeddings is None or query_embedding is None:
        return []
    
    try:
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
        # Critical check: Are embeddings loaded?
        if product_embeddings is None:
            return jsonify({
                'success': False,
                'message': 'ResNet50 embeddings not loaded. Please run feature_extractor.py first.'
            })
        
        # Get image path from request
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'success': False, 'message': 'No image path provided'})
        
        image_path = data['image_path']
        
        # Build full path to uploaded image
        if image_path.startswith('uploads/'):
            full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path.replace('uploads/', ''))
        else:
            full_image_path = os.path.join('static', image_path)
        
        # Check if file exists
        if not os.path.exists(full_image_path):
            return jsonify({'success': False, 'message': f'Image file not found: {image_path}'})
        
        # Process image and find similar products
        print(f"🔍 Processing image with ResNet50: {full_image_path}")
        query_embedding = enhanced_preprocess_image_for_search(full_image_path)
        
        if query_embedding is None:
            return jsonify({'success': False, 'message': 'Failed to process uploaded image'})
        
        # Find similar products
        similar_products = find_similar_products_enhanced(query_embedding, top_k=12)
        
        if not similar_products:
            return jsonify({
                'success': True,
                'results': [],
                'message': 'No similar products found'
            })
        
        print(f"✅ Found {len(similar_products)} similar products")
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
    """Health check endpoint"""
    model_status = "✅ Ready" if feature_model is not None else "❌ Not loaded"
    embeddings_status = "✅ Ready" if product_embeddings is not None else "❌ Not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'embeddings': embeddings_status,
        'products': len(product_embeddings) if product_embeddings is not None else 0
    })

if __name__ == '__main__':
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRODUCT_FOLDER'], exist_ok=True)
    
    # Load embeddings and model on startup
    load_embeddings_and_model()
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌐 Server starting on port {port}")
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
