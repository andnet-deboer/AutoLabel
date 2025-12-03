import os
import shutil
import argparse
import random
import yaml
import cv2
import torch
import sys
import numpy as np
from PIL import Image

# SAM2 Imports
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Grounding DINO Imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def masks_to_yolo_format(masks, img_shape, class_id):
    h, w = img_shape
    yolo_boxes = []
    
    for mask in masks:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            continue
            
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        x_center = x_min + box_w / 2
        y_center = y_min + box_h / 2
        
        yolo_boxes.append([
            class_id,
            x_center / w,
            y_center / h,
            box_w / w,
            box_h / h
        ])
    return yolo_boxes

def mask_to_obb(mask, img_shape, class_id):
    h, w = img_shape

    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None

    pts = np.vstack((xs, ys)).T.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)

    box_norm = []
    for x, y in box:
        box_norm.append(x / w)
        box_norm.append(y / h)

    return [class_id] + box_norm

def save_yolo_obb(output_path, obb_list):
    with open(output_path, "w") as f:
        for obb in obb_list:
            f.write(" ".join(map(str, obb)) + "\n")

def save_yolo_labels(output_path, yolo_boxes):
    with open(output_path, "w") as f:
        for box in yolo_boxes:
            f.write(" ".join(map(str, box)) + "\n")

def create_dataset_yaml(output_dir, class_names):
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Generated config: {yaml_path}")

def load_prompt_map(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Prompt config file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def process_dataset(input_dir, output_dir, prompt_config_path, bbox_format,
                    box_threshold=0.35, device="cuda", split_ratio=0.8):

    # Setup directories
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for sd in subdirs:
        os.makedirs(os.path.join(output_dir, sd), exist_ok=True)

    prompt_map = load_prompt_map(prompt_config_path)
    print(f"Loaded {len(prompt_map)} class prompts.")

    # Load models
    print(f"Loading Grounding DINO on {device}...")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)

    print(f"Loading SAM 2 on {device}...")
    sam_model = build_sam2_hf("facebook/sam2-hiera-base-plus", device=device)
    predictor = SAM2ImagePredictor(sam_model)

    # Map classes
    class_names = sorted([d for d in os.listdir(input_dir)
                          if os.path.isdir(os.path.join(input_dir, d))])
    class_to_id = {name: i for i, name in enumerate(class_names)}

    all_samples = []
    print(f"Scanning directory (taking every 10th image)...")

    for class_name in class_names:
        rgb_folder = os.path.join(input_dir, class_name, "rgb")
        if not os.path.exists(rgb_folder):
            continue

        # Prompt lookup
        current_prompt = prompt_map.get(class_name, class_name.replace("_", " "))
        if not current_prompt.endswith("."):
            current_prompt += "."

        print(f"  > Class '{class_name}' -> Prompt: '{current_prompt}'")

        files = sorted([f for f in os.listdir(rgb_folder)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        subset = files[::10]

        for fname in subset:
            all_samples.append({
                'src_path': os.path.join(rgb_folder, fname),
                'filename': f"{class_name}_{fname}",
                'class_id': class_to_id[class_name],
                'class_name': class_name,
                'text_prompt': current_prompt
            })

    # Shuffle & split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * split_ratio)
    sets = {'train': all_samples[:split_idx],
            'val': all_samples[split_idx:]}

    # Processing loop
    for split_name, samples in sets.items():
        print(f"Processing {split_name} set ({len(samples)} images)...")

        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"  {i}/{len(samples)}...")

            image_cv = cv2.imread(sample['src_path'])
            if image_cv is None:
                continue

            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Grounding DINO
            inputs = processor(images=pil_image, text=sample['text_prompt'],
                               return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = dino_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=box_threshold,
                target_sizes=[pil_image.size[::-1]]
            )

            dino_boxes = results[0]["boxes"].cpu().numpy()
            if len(dino_boxes) == 0:
                continue

            # SAM segmentation
            predictor.set_image(image_rgb)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=dino_boxes,
                multimask_output=False
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            # Select output format
            h, w = image_rgb.shape[:2]

            if bbox_format == "yolo":
                yolo_boxes = masks_to_yolo_format(masks, (h, w), sample['class_id'])
                if not yolo_boxes:
                    continue
                label_data = yolo_boxes
                save_fn = save_yolo_labels

            else:  # OBB
                obb_boxes = []
                for mask in masks:
                    obb = mask_to_obb(mask, (h, w), sample['class_id'])
                    if obb:
                        obb_boxes.append(obb)
                if not obb_boxes:
                    continue
                label_data = obb_boxes
                save_fn = save_yolo_obb

            # Save image
            dest_img_path = os.path.join(output_dir, 'images', split_name, sample['filename'])
            shutil.copy2(sample['src_path'], dest_img_path)

            # Save label
            label_name = os.path.splitext(sample['filename'])[0] + ".txt"
            dest_label_path = os.path.join(output_dir, 'labels', split_name, label_name)
            save_fn(dest_label_path, label_data)

    create_dataset_yaml(output_dir, class_names)
    print("\nProcessing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", required=True, help="Path to prompts.yaml file")
    parser.add_argument("--bbox-format", choices=["yolo", "obb"], default="yolo",
                        help="Choose bounding box format: 'yolo' or 'obb'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    process_dataset(args.input, args.output,
                    prompt_config_path=args.prompts,
                    bbox_format=args.bbox_format,
                    device=args.device)
