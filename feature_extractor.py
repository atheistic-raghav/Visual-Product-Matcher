#!/usr/bin/env python3
# CPU-only MobileNetV2 embedding generator compatible with app.py

# 1) Hide all GPUs BEFORE importing TensorFlow to avoid cuInit/XLA/CUDA attempts and errors.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Force CPU-only runtime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # Reduce TensorFlow INFO/WARNING logs

import sys
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# --- Configuration ---
PRODUCTS_FOLDER = Path('data/products')
CSV_PATH        = Path('data/products.csv')
EMBEDDINGS_PATH = Path('data/product_embeddings.npz')

BATCH_SIZE   = 16
TARGET_SIZE  = (224, 224)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}


def load_model():
    """Load MobileNetV2 feature extractor (CPU-only) with correct 224x224x3 input shape."""
    print("🚀 Loading MobileNetV2 model (ImageNet weights)...")
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)   # FIXED: Was (TARGET_SIZE, ...), now (224, 224, 3)
    )
    model = Model(inputs=base.input, outputs=base.output)
    # compile() is not required for inference but is harmless
    model.compile()
    print(f"✅ MobileNetV2 loaded with {model.count_params():,} parameters")
    return model

def preprocess_image(img_path: Path):
    """Match app.py preprocessing for consistent embeddings."""
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if img.width < 64 or img.height < 64:
                print(f"⚠️ Image too small: {img_path}")
                return None
            # Aspect-ratio-preserving fit and center-crop (matches app.py)
            img = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)
            # Light denoise + enhancements (matches app.py)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = ImageEnhance.Sharpness(img).enhance(1.1)
            img = ImageEnhance.Contrast(img).enhance(1.05)
            img = ImageEnhance.Color(img).enhance(1.03)
            # Convert to array and apply MobileNetV2 preprocessing
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            return arr
    except Exception as e:
        print(f"❌ Error preprocessing {img_path}: {e}")
        return None

def extract_batch_embeddings(model: Model, batch_imgs: list):
    """Run a batch through the model and L2-normalize the output embeddings."""
    try:
        batch = np.vstack(batch_imgs)
        features = model.predict(batch, verbose=0)
        # Normalize embeddings to unit vectors for consistent cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.clip(norms, 1e-8, None)
        return features
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return None

def validate_inputs():
    """Ensure CSV and product images directory exist and contain images."""
    if not CSV_PATH.exists():
        print(f"❌ CSV file missing: {CSV_PATH}")
        return False
    if not PRODUCTS_FOLDER.exists():
        print(f"❌ Products folder missing: {PRODUCTS_FOLDER}")
        return False
    
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(PRODUCTS_FOLDER.glob(f'*{ext}'))
        image_files.extend(PRODUCTS_FOLDER.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print("❌ No images found in products folder.")
        return False
    return True

def main():
    """Main pipeline: load metadata, build model, preprocess, embed, and save."""
    if not validate_inputs():
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    model = load_model()

    embeddings, filenames, names, categories = [], [], [], []
    batch_imgs, batch_meta = [], []

    print(f"🎯 Processing {len(df)} images with batch size {BATCH_SIZE}...")
    for idx, row in df.iterrows():
        img_path = PRODUCTS_FOLDER / row['filename']
        if not img_path.exists():
            print(f"⚠️ Missing file: {img_path}")
            continue

        arr = preprocess_image(img_path)
        if arr is None:
            continue

        batch_imgs.append(arr)
        batch_meta.append(row)

        if len(batch_imgs) == BATCH_SIZE or idx == len(df) - 1:
            batch_embeddings = extract_batch_embeddings(model, batch_imgs)
            if batch_embeddings is not None:
                for emb, meta in zip(batch_embeddings, batch_meta):
                    embeddings.append(emb)
                    filenames.append(meta['filename'])
                    names.append(meta['name'])
                    categories.append(meta['category'])
            
            batch_imgs.clear()
            batch_meta.clear()
            gc.collect()

    # Save compressed NPZ file compatible with app.py
    embeddings = np.array(embeddings, dtype=np.float32)
    filenames  = np.array(filenames)
    names      = np.array(names)
    categories = np.array(categories)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        filenames=filenames,
        names=names,
        categories=categories,
        extraction_info={
            'model': 'MobileNetV2',
            'embedding_dim': embeddings.shape[1] if embeddings.size > 0 else 0,
            'total_images': len(embeddings),
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'preprocessing': 'enhanced_mobilenet'
        }
    )
    print(f"💾 Saved embeddings ({embeddings.shape}) to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    main()
