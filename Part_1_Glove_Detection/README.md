# Gloved vs Ungloved Hand Detection (YOLOv8)

## Problem Statement
Build a computer vision system to detect whether workers are wearing gloves
(gloved hand vs bare hand) from images captured in factory environments.

## Model
- YOLOv8n (Ultralytics)

## Dataset
- Source: Roboflow Universe
- Classes:
  - Glove
  - No Glove
- Format: YOLO
- Split: Train / Validation / Test

## Training Details
- Image size: 640x640
- Epochs: 20+
- Device: CPU
- Optimizer: AdamW (auto)

## Inference
- Multiple hands detected per image
- Bounding boxes with confidence scores
- Annotated outputs saved as images
- JSON logs generated per image

## Results
Final test predictions are available in:

output/annotated_images/

## Limitations
- Missed detections may occur due to blur, occlusion, or low confidence
- This reflects real-world conditions in factory environments

## How to Run
```bash
pip install ultralytics opencv-python
python detection_script.py --input dataset/test/images --output output --logs logs
