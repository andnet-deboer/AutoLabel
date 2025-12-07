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
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION: MAP FOLDERS TO OUTPUT CLASSES ---
FOLDER_MAPPING = {
    "db_class_191":    ["db_class_191"],
    "40ft_reefer":     ["40ft_reefer"],
    "nyc_caboose":     ["nyc_caboose"],
    "kato_controller": ["direction_switch", "speed_knob"]
}

# Automatically generate the master list of ID numbers for YOLO
YOLO_LABELS = []
for k, v in FOLDER_MAPPING.items():
    for label in v:
        if label not in YOLO_LABELS:
            YOLO_LABELS.append(label)

print(f"Global Class Registry: {YOLO_LABELS}")

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

def save_labels(output_path, data):
    with open(output_path, "w") as f:
        for item in data:
            f.write(" ".join(map(str, item)) + "\n")

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
                    threshold=0.35, device="cuda", split_ratio=0.8,
                    sample_rate=10, use_yolo=False, yolo_model_path=None):

    # Setup directories
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for sd in subdirs:
        os.makedirs(os.path.join(output_dir, sd), exist_ok=True)

    prompt_map = load_prompt_map(prompt_config_path)

    # --- MODEL LOADING ---
    dino_processor = None
    dino_model = None
    yolo_model = None

    print(f"Loading SAM 2 on {device}...")
    sam_model = build_sam2_hf("facebook/sam2-hiera-base-plus", device=device)
    predictor = SAM2ImagePredictor(sam_model)

    if use_yolo:
        print(f"Loading YOLO model from {yolo_model_path}...")
        try:
            yolo_model = YOLO(str(yolo_model_path))
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
    else:
        print(f"Loading Grounding DINO on {device}...")
        dino_id = "IDEA-Research/grounding-dino-tiny"
        dino_processor = AutoProcessor.from_pretrained(dino_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)

    # Gather images
    all_samples = []

    for folder_name, output_classes in FOLDER_MAPPING.items():
        rgb_folder = os.path.join(input_dir, folder_name, "rgb")
        if not os.path.exists(rgb_folder): continue

        # Prompt setup (Even if using YOLO, we check logic)
        raw_prompt = prompt_map.get(folder_name, "")
        
        # Determine Multi-Class Logic
        is_multi_class = len(output_classes) > 1
        prompt_parts = []
        text_prompt = ""

        if not use_yolo:
            # Prepare DINO prompts
            if isinstance(raw_prompt, list):
                text_prompt = ". ".join(raw_prompt)
            else:
                text_prompt = raw_prompt or ""
            
            if not text_prompt.endswith("."): text_prompt += "."
            
            if is_multi_class:
                prompt_parts = [p.strip().lower().replace(".", "") for p in text_prompt.split('.') if p.strip()]

        files = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        subset = files[::sample_rate]

        for fname in subset:
            all_samples.append({
                'src_path': os.path.join(rgb_folder, fname),
                'filename': f"{folder_name}_{fname}",
                'text_prompt': text_prompt,
                'output_classes': output_classes,
                'is_multi_class': is_multi_class,
                'prompt_parts': prompt_parts
            })

    # Shuffle & split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * split_ratio)
    sets = {'train': all_samples[:split_idx], 'val': all_samples[split_idx:]}

    for split_name, samples in sets.items():
        print(f"Processing {split_name} set ({len(samples)} images)...")

        for i, sample in enumerate(samples):
            if i % 10 == 0: print(f"  {i}/{len(samples)}...")

            image_cv = cv2.imread(sample['src_path'])
            if image_cv is None: continue
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            h, w = image_rgb.shape[:2]

            candidate_boxes = [] # List of (box, class_id)

            # ==============================
            # OPTION A: YOLO INFERENCE
            # ==============================
            if use_yolo:
                results = yolo_model(image_rgb, verbose=False)[0]
                
                # We use OBB if available, otherwise standard boxes
                raw_boxes = results.obb if results.obb is not None else results.boxes
                
                if len(raw_boxes) > 0:
                    for j, box in enumerate(raw_boxes):
                        # Get box coordinates (x,y,x,y)
                        # Ensure we convert to numpy [1, 4]
                        b_xyxy = box.xyxy.cpu().numpy().reshape(1, 4)
                        
                        # --- CLASS ASSIGNMENT ---
                        # Challenge: YOLO gives us a class index 'j', but it might not match our new YAML
                        # Strategy:
                        # 1. If Single-Class Folder: IGNORE YOLO class, force folder class.
                        # 2. If Multi-Class Folder: We default to the first class in the list
                        #    (Because we can't distinguish levers/knobs without DINO text)
                        
                        target_class_name = sample['output_classes'][0]
                        try:
                            cid = YOLO_LABELS.index(target_class_name)
                            candidate_boxes.append((b_xyxy, cid))
                        except ValueError: continue

            # ==============================
            # OPTION B: GROUNDING DINO
            # ==============================
            else:
                inputs = dino_processor(images=pil_image, text=sample['text_prompt'], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = dino_model(**inputs)

                results = dino_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    threshold=threshold, text_threshold=threshold,
                    target_sizes=[pil_image.size[::-1]]
                )

                dino_boxes = results[0]["boxes"].cpu().numpy()
                dino_phrases = results[0]["labels"]

                for box, phrase in zip(dino_boxes, dino_phrases):
                    target_class_name = None
                    
                    if sample['is_multi_class']:
                        phrase_clean = phrase.lower().strip()
                        for idx, part in enumerate(sample['prompt_parts']):
                            if part in phrase_clean or phrase_clean in part:
                                target_class_name = sample['output_classes'][idx]
                                break
                    else:
                        target_class_name = sample['output_classes'][0]

                    if target_class_name:
                        try:
                            cid = YOLO_LABELS.index(target_class_name)
                            candidate_boxes.append((box.reshape(1, 4), cid))
                        except ValueError: continue

            if not candidate_boxes: continue

            # ==============================
            # SHARED: SAM2 SEGMENTATION
            # ==============================
            final_labels = []
            
            for (box, class_id) in candidate_boxes:
                predictor.set_image(image_rgb)
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                if masks.ndim == 4: masks = masks.squeeze(1)

                if bbox_format == "yolo":
                    yolo_boxes = masks_to_yolo_format(masks, (h, w), class_id)
                    final_labels.extend(yolo_boxes)
                else:
                    for m in masks:
                        obb = mask_to_obb(m, (h, w), class_id)
                        if obb: final_labels.append(obb)

            if not final_labels: continue

            # Save
            dest_img_path = os.path.join(output_dir, 'images', split_name, sample['filename'])
            shutil.copy2(sample['src_path'], dest_img_path)

            label_name = os.path.splitext(sample['filename'])[0] + ".txt"
            dest_label_path = os.path.join(output_dir, 'labels', split_name, label_name)
            save_labels(dest_label_path, final_labels)

    create_dataset_yaml(output_dir, YOLO_LABELS)
    print("\nProcessing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--bbox-format", choices=["yolo", "obb"], default="yolo")
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # YOLO Specific Args
    parser.add_argument("--yolo", action="store_true", help="Use YOLO model instead of Grounding DINO")
    parser.add_argument("--model-path", type=str, 
                        default=str(Path(__file__).resolve().parent / "models" / "train_topview_obb.pt"),
                        help="Path to .pt model file for YOLO")
    
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.prompts, args.bbox_format, 
                    sample_rate=args.sample_rate, device=args.device,
                    use_yolo=args.yolo, yolo_model_path=args.model_path)