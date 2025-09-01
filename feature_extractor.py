#!/usr/bin/env python3

import os
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

# Configuration
PRODUCTS_FOLDER = Path('data/products')
CSV_PATH        = Path('data/products.csv')
EMBEDDINGS_PATH = Path('data/product_embeddings.npz')

BATCH_SIZE    = 16
TARGET_SIZE   = (224, 224)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}

def load_model():
    """Load MobileNetV2 feature extractor (CPU-only)."""
    print("🚀 Loading MobileNetV2 model (ImageNet weights)...")
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)
    )
    model = Model(inputs=base.input, outputs=base.output)
    model.compile()
    print(f"✅ MobileNetV2 loaded with {model.count_params():,} parameters")
    return model

def preprocess_image(img_path: Path):
    """Match app.py preprocessing: resize, enhancements, MobileNetV2 input."""
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if img.width < 64 or img.height < 64:
                print(f"⚠️ Image too small: {img_path}")
                return None

            # Preserve aspect ratio, then center-crop to TARGET_SIZE
            img = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)

            # Denoise + enhancements
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = ImageEnhance.Sharpness(img).enhance(1.1)
            img = ImageEnhance.Contrast(img).enhance(1.05)
            img = ImageEnhance.Color(img).enhance(1.03)

            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            return arr
    except Exception as e:
        print(f"❌ Error preprocessing {img_path}: {e}")
        return None

def extract_batch_embeddings(model: Model, batch_imgs: list):
    """Run batch through model and normalize outputs."""
    try:
        batch = np.vstack(batch_imgs)
        features = model.predict(batch, verbose=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, 1e-8, None)
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return None

def validate_inputs():
    """Ensure CSV and product images directory exist."""
    if not CSV_PATH.exists():
        print(f"❌ CSV file missing: {CSV_PATH}")
        return False
    if not PRODUCTS_FOLDER.exists():
        print(f"❌ Products folder missing: {PRODUCTS_FOLDER}")
        return False

    imgs = []
    for ext in SUPPORTED_FORMATS:
        imgs.extend(PRODUCTS_FOLDER.glob(f'*{ext}'))
        imgs.extend(PRODUCTS_FOLDER.glob(f'*{ext.upper()}'))
    if not imgs:
        print("❌ No images found in products folder.")
        return False
    return True

def main():
    """Main pipeline: validate, load model, process in batches, save embeddings."""
    if not validate_inputs():
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    model = load_model()

    embeddings = []
    filenames  = []
    names      = []
    categories = []
    batch_imgs = []
    batch_meta = []

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

        # Process when batch full or last image
        if len(batch_imgs) == BATCH_SIZE or idx == len(df) - 1:
            batch_embs = extract_batch_embeddings(model, batch_imgs)
            if batch_embs is not None:
                for emb, meta in zip(batch_embs, batch_meta):
                    embeddings.append(emb)
                    filenames.append(meta['filename'])
                    names.append(meta['name'])
                    categories.append(meta['category'])
            batch_imgs.clear()
            batch_meta.clear()
            gc.collect()

    embeddings = np.array(embeddings, dtype=np.float32)
    filenames  = np.array(filenames)
    names      = np.array(names)
    categories = np.array(categories)

    EMB_DIR = EMBEDDINGS_PATH.parent
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        filenames=filenames,
        names=names,
        categories=categories,
        extraction_info={
            'model': 'MobileNetV2',
            'embedding_dim': embeddings.shape[1],
            'total_images': len(embeddings),
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'preprocessing': 'enhanced_mobilenet'
        }
    )
    print(f"💾 Saved embeddings ({embeddings.shape}) to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    main()
