import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import gc

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

PRODUCTS_FOLDER = 'data/products'
CSV_PATH = 'data/products.csv'
EMBEDDINGS_PATH = 'data/product_embeddings.npz'

print("🚀 Running MobileNetV2 based Feature Extractor")

# Load product metadata
try:
    df = pd.read_csv(CSV_PATH)
    print(f"📊 Loaded {len(df)} products from {CSV_PATH}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit(1)

def load_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)

def enhanced_preprocess_image(img_path):
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
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
        print(f"❌ Error preprocessing image {img_path}: {e}")
        return None

def extract_embeddings():
    model = load_model()
    embeddings = []
    filenames = []
    names = []
    categories = []

    print(f"📦 Extracting embeddings for {len(df)} products...")
    
    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        img_path = os.path.join(PRODUCTS_FOLDER, row['filename'])
        if not os.path.exists(img_path):
            print(f"❌ Missing image file {row['filename']}")
            failed += 1
            continue
        img_arr = enhanced_preprocess_image(img_path)
        if img_arr is None:
            print(f"❌ Failed to process {row['filename']}")
            failed += 1
            continue
        embedding = model.predict(img_arr, verbose=0)[0]
        embeddings.append(embedding)
        filenames.append(row['filename'])
        names.append(row['name'])
        categories.append(row['category'])
        successful += 1
        if successful % 10 == 0:
            gc.collect()
            print(f"🧹 Processed {successful} images, cleaning memory")

    print(f"✅ Successfully processed: {successful}")
    print(f"⚠️ Failed: {failed}")

    if successful == 0:
        print("❌ No embeddings extracted, aborting.")
        return
    
    embeddings = np.array(embeddings)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        filenames=np.array(filenames),
        names=np.array(names),
        categories=np.array(categories)
    )

    file_size_mb = os.path.getsize(EMBEDDINGS_PATH) / (1024 * 1024)
    print(f"💾 Saved embeddings to {EMBEDDINGS_PATH}, size {file_size_mb:.2f} MB")

if __name__ == '__main__':
    if not os.path.exists(PRODUCTS_FOLDER):
        print(f"❌ Products folder not found: {PRODUCTS_FOLDER}")
        exit(1)
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV metadata file not found: {CSV_PATH}")
        exit(1)
    
    extract_embeddings()
