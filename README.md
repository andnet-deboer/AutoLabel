<p align="center">
  <img src="assets/overview.png" width="100%" />
</p>

# Auto-Labeling & YOLO/OBB Dataset Pipeline

Automatically generate bounding box or oriented box labels for object detection datasets using **Grounding DINO** and **SAM2**.

---

## Quick Start


### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clone and install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..

# 3. Auto-label your images
python auto_label.py \
    --input ./yourdata \
    --output ./yolo_dataset \
    --prompts prompts.yaml \
    --bbox-format yolo

# 4. Train YOLOv8
yolo detect train data=yolo_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

---

## Project Files

| File | Description |
|------|-------------|
| `auto_label.py` | Main auto-labeling script. Generates YOLO or OBB labels from images. |
| `prompts.yaml` | Text prompts corresponding to each class. Required for Grounding DINO. |
| `trains/` | Raw images organized by class (e.g., `trains/class_name/rgb/*.jpg`). |
| `yolo_dataset/` | Output folder for generated dataset (images + labels + `dataset.yaml`). |
| `view_dataset.py` | Optional visualization tool for verifying labels. |

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch + CUDA (recommended)
- OpenCV
- Transformers (for Grounding DINO)
- SAM2

### SAM2 Setup

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

---

## Auto-Labeling Usage

```bash
python auto_label.py \
    --input ./trains \
    --output ./yolo_dataset \
    --prompts prompts.yaml \
    --bbox-format yolo
```

### Options

| Option | Description |
|--------|-------------|
| `--bbox-format yolo` | Axis-aligned YOLO boxes (`x_center, y_center, width, height` normalized) |
| `--bbox-format obb` | Rotated oriented bounding boxes |
| `--device cuda` | Use GPU if available (defaults to CPU) |

### Workflow

1. **Grounding DINO** generates rough object bounding boxes from text prompts
2. **SAM2** refines these boxes into precise segmentation masks
3. Labels are saved in YOLO or OBB format
4. Images are copied to `images/train` and `images/val` with matching labels in `labels/`

---

## Dataset Structure

After running `auto_label.py`, the output directory looks like:

```
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

The `dataset.yaml` is auto-generated for YOLO/Ultralytics training:

```yaml
path: /absolute/path/to/yolo_dataset
train: images/train
val: images/val
nc: <number_of_classes>
names:
  0: class_name_0
  1: class_name_1
  ...
```

---

## Visualizing Generated Labels

Verify your auto-labeled dataset:

```bash
python view_dataset.py --data ./yolo_dataset --prompts ./prompts.yaml
```

---

## Training YOLO

### Install Ultralytics

```bash
pip install ultralytics
```

### Train a YOLOv8 Model

```bash
yolo detect train \
    data=yolo_dataset/dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640
```

| Model | Description |
|-------|-------------|
| `yolov8n.pt` | Nano (fastest, least accurate) |
| `yolov8s.pt` | Small |
| `yolov8m.pt` | Medium |
| `yolov8l.pt` | Large |
| `yolov8x.pt` | Extra Large (slowest, most accurate) |

Training outputs are saved to `runs/detect/train/`.

---

## Notes & Tips

- Ensure `prompts.yaml` matches all class names in your dataset
- **YOLO format** → axis-aligned bounding boxes
- **OBB format** → rotated boxes for better alignment on non-rectangular objects
- By default, only every 10th image is processed to reduce runtime; adjust `subset = files[::10]` in `auto_label.py` if needed
- **GPU acceleration is highly recommended** for SAM2 and Grounding DINO

---

## License

MIT License - feel free to use and modify for your projects.