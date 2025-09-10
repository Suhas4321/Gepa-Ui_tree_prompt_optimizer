# utils/dataset_loader.py

import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split

def load_image(image_path):
    with Image.open(image_path) as img:
        return img.copy()

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataset(images_dir, json_dir):
    dataset = []
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Screenshots directory not found: {images_dir}")
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON tree directory not found: {json_dir}")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    json_files = sorted([f for f in os.listdir(json_dir) if f.lower().endswith('json')])
    
    image_map = {os.path.splitext(f)[0]: f for f in image_files}
    json_map = {os.path.splitext(f)[0]: f for f in json_files}
    
    common_names = set(image_map.keys()) & set(json_map.keys())
    
    for name in sorted(common_names):
        try:
            image_path = os.path.join(images_dir, image_map[name])
            json_path = os.path.join(json_dir, json_map[name])
            
            dataset.append({
                'image': load_image(image_path),
                'expected_ui_tree': load_json(json_path),
                'filename': name
            })
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue
    
    return dataset

def split_dataset(dataset, train_ratio, val_ratio, random_state, **kwargs):
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio and val_ratio must sum to 1.0")
    
    trainset, valset = train_test_split(
        dataset, 
        test_size=val_ratio, 
        random_state=random_state
    )
    
    return trainset, valset