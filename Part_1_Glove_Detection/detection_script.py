import os
import json
import cv2
import argparse
from ultralytics import YOLO


def run_detection(input_dir, output_dir, log_dir, conf_thresh):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    
    model = YOLO("model/best.pt")


    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        
        results = model(
            image,
            conf=conf_thresh,
            iou=0.7,
            max_det=50,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = "gloved_hand" if cls == 0 else "bare_hand"
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

                color = (0, 255, 0) if label == "gloved_hand" else (0, 0, 255)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        
        cv2.imwrite(os.path.join(output_dir, img_name), image)

        
        json_name = os.path.splitext(img_name)[0] + ".json"
        log_data = {
            "filename": img_name,
            "detections": detections
        }

        with open(os.path.join(log_dir, json_name), "w") as f:
            json.dump(log_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glove vs Bare Hand Detection")
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", default="output", help="Output folder for images")
    parser.add_argument("--logs", default="logs", help="Output folder for JSON logs")
    parser.add_argument("--confidence", type=float, default=0.05, help="Confidence threshold")

    args = parser.parse_args()

    run_detection(args.input, args.output, args.logs, args.confidence)
