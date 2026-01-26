import pandas as pd
import numpy as np
from PIL import Image, ImageChops
import os
from shutil import rmtree, copyfile
from tqdm import tqdm

# --- Configuration ---
IMAGE_FOLDER = 'images/image_train/'
CLEANUP_CSV = 'images_to_remove.csv'
X_TRAIN_UPDATE = 'X_train_update.csv'
TRAIN_CSV = 'train_2.csv'
TEST_CSV = 'test_2.csv'

IMAGE_SCALED_FOLDER = 'images/image_train_scaled_15/'
TRAIN_OUT_FOLDER = 'image_train_15'
TEST_OUT_FOLDER = 'image_test_15'

TARGET_SIZE = (224, 224)
PADDING = 0
MIN_BB_AREA = 15000  # Minimum bounding box area to process the image
MAX_ASPECT_RATIO = 3.0 # Maximum aspect ratio to keep small images

def get_bbox(im):
    """
    Computes the bounding box of the foreground object.
    Assumes the background color is the color of the pixel at (0,0).
    """
    if im is None:
        return (0, 0, 0, 0)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        # If the image is a solid color, use the whole image
        return (0, 0, im.size[0], im.size[1])
    return bbox

def crop_image_square(image, x1, y1, x2, y2, padding=0):
    """
    Crops the image to a square area containing the bounding box.
    Logic strictly matched from cropImage in 26_ImagePreproc.ipynb.
    """
    w, h = image.size
    
    # Apply padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(x2 + padding, w)
    y2 = min(y2 + padding, h)

    bw = x2 - x1
    bh = y2 - y1

    if bw == w:
        y1 = 0
        y2 = h
    if bh == h:
        x1 = 0
        x2 = w

    # Make it square
    if bw > bh:
        d = bw - bh
        y1 = max(0, y1 - d // 2)
        y2 = min(y2 + d // 2, h)
        
        bw, bh = x2 - x1, y2 - y1
        d = bw - bh

        if bw - bh > 1:
            if y1 == 0:
                y2 = min(y2 + d, h)
            elif y2 == h:
                y1 = max(y1 - d, 0)
            else:
                y1 = max(0, y1 - d // 2)
                y2 = min(y2 + d // 2, h)

        bw, bh = x2 - x1, y2 - y1
        d = bw - bh

        if bw - bh == 1:
            if y1 > 0:
                y1 -= 1
            elif y2 < h:
                y2 += 1

    elif bh > bw:
        d = bh - bw
        x1 = max(0, x1 - d // 2)
        x2 = min(x2 + d // 2, w)
        
        bw, bh = x2 - x1, y2 - y1
        d = bh - bw

        if bh - bw > 1:
            if x1 == 0:
                x2 = min(x2 + d, w)
            elif x2 == w:
                x1 = max(x1 - d, 0)
            else:
                x1 = max(0, x1 - d // 2)
                x2 = min(x2 + d // 2, w)

        bw, bh = x2 - x1, y2 - y1
        d = bh - bw

        if bh - bw == 1:
            if x1 > 0:
                x1 -= 1
            elif x2 < w:
                x2 += 1

    return image.crop((x1, y1, x2, y2))

def step1_scale_images():
    print("--- Step 1: Scaling and Cropping Images ---")
    if os.path.exists(IMAGE_SCALED_FOLDER):
        print(f"Removing existing folder: {IMAGE_SCALED_FOLDER}")
        rmtree(IMAGE_SCALED_FOLDER)
    os.makedirs(IMAGE_SCALED_FOLDER)

    # Use X_train_update.csv as the source for image list
    df = pd.read_csv(X_TRAIN_UPDATE, index_col=0)

    count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scaling images"):
        im_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        img_path = os.path.join(IMAGE_FOLDER, im_name)
        
        if not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as img:
                # Compute bounding box on the fly
                bbox = get_bbox(img)
                x1, y1, x2, y2 = bbox
                bw = x2 - x1
                bh = y2 - y1
                bb_area = bw * bh
                
                # skip small images unless they have extreme aspect ratio
                if bb_area < MIN_BB_AREA:
                    # aspect_ratio = bw / bh if bh > 0 else 0
                    # if aspect_ratio < MAX_ASPECT_RATIO and aspect_ratio > 1./MAX_ASPECT_RATIO:
                    #     # Skip this small image
                    #     continue
                    continue

                img_cropped = crop_image_square(img, x1, y1, x2, y2, PADDING)
                img_resized = img_cropped.resize(TARGET_SIZE, Image.LANCZOS)
                img_resized.save(os.path.join(IMAGE_SCALED_FOLDER, im_name))
                count += 1
        except Exception as e:
            print(f"Error processing {im_name}: {e}")

    print(f"Finished Step 1. Processed {count} images.")

def step2_cleanup_images():
    print("\n--- Step 2: Cleaning up Scaled Folder ---")
    if not os.path.exists(CLEANUP_CSV):
        print(f"Cleanup file {CLEANUP_CSV} not found. Skipping Step 2.")
        return

    cleanup_df = pd.read_csv(CLEANUP_CSV)
    removed = 0
    for _, row in tqdm(cleanup_df.iterrows(), total=len(cleanup_df), desc="Removing duplicates/artifacts"):
        filename = row['filename']
        file_path = os.path.join(IMAGE_SCALED_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            removed += 1
    
    print(f"Finished Step 2. Removed {removed} images.")

def step3_organize_by_category():
    print("\n--- Step 3: Organizing into Train/Test folders by category ---")
    
    def process_split(split_csv, out_folder, desc):
        if os.path.exists(out_folder):
            rmtree(out_folder)
        os.makedirs(out_folder)
        
        df = pd.read_csv(split_csv, index_col=0)
        skipped = 0
        copied = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            im_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"
            category = str(row['prdtypecode'])
            
            src_path = os.path.join(IMAGE_SCALED_FOLDER, im_name)
            if os.path.exists(src_path):
                cat_folder = os.path.join(out_folder, category)
                if not os.path.exists(cat_folder):
                    os.makedirs(cat_folder)
                copyfile(src_path, os.path.join(cat_folder, im_name))
                copied += 1
            else:
                skipped += 1
        print(f"{desc} complete. Copied: {copied}, Skipped (missing or removed): {skipped}")

    process_split(TRAIN_CSV, TRAIN_OUT_FOLDER, "Organizing Train set")
    process_split(TEST_CSV, TEST_OUT_FOLDER, "Organizing Test set")

if __name__ == "__main__":
    step1_scale_images()
    step2_cleanup_images()
    step3_organize_by_category()
