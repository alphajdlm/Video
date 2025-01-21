# Install Roboflow if not already installed
#!pip install roboflow

import os
import cv2
import numpy as np
from flask import Flask, Response, request
from roboflow import Roboflow
from ultralytics import YOLO
import time

# Roboflow dataset integration
rf = Roboflow(api_key="64sVoTpUQkh1K7MlJEnm")
project = rf.workspace("bhanu-prakash-eucim").project("person-detector-fhl7g")
version = project.version(1)

try:
    # Download dataset and get its location
    dataset = version.download("yolov8")
    dataset_path = dataset.location

    if not os.path.exists(os.path.join(dataset_path, "train")):
        raise FileNotFoundError(f"'train' folder not found in {dataset_path}")
except Exception as e:
    print(f"Error downloading or accessing dataset: {e}")
    dataset_path = None

# Load YOLO model
model = YOLO('yolov8x.pt')

# Video sources with unique IDs
video_sources = {
    "1": "CP4G019985-2025-01-12-12-01-51.mp4",
    "2": "CP4G019985-2025-01-12-14-19-47.mp4",
    "3": "CP4G019985-2025-01-12-16-11-03.mp4",
    "4": "CP4G011555-2025-01-12-12-18-32.mp4",
    "5": "CP4G011555-2025-01-12-15-41-51.mp4",
    "6": "CP4G147289-2025-01-12-16-10-40.mp4",
    "7": "CP4G147289-2025-01-12-16-25-38.mp4",
    "8": "CP4G147289-2025-01-12-14-10-53.mp4",
    "9": "CP4G147289-2025-01-12-11-12-23.mp4",
    "10": "CP4G147289-2025-01-12-14-25-51.mp4",
    "11": "rtsp://192.168.0.101:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
    "12": "rtsp://192.168.0.58:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
}

# Add Roboflow dataset images as sources if the dataset was successfully loaded
if dataset_path:
    for idx, image_file in enumerate(os.listdir(os.path.join(dataset_path, "train")), start=13):
        if image_file.endswith(('.jpg', '.png')):
            video_sources[str(idx)] = os.path.join(dataset_path, "train", image_file)

# Load class labels
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Flask app initialization
app = Flask(__name__)

def apply_nms(boxes, scores, iou_threshold=0.4):
    """
    Apply Non-Maximum Suppression to eliminate overlapping bounding boxes.
    """
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_area = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        union_area = areas[i] + areas[order[1:]] - inter_area
        iou = inter_area / union_area

        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]

    return keep

def generate_frames(video_id):
    """Generate video frames for the given video ID with normal frame rate."""
    if video_id not in video_sources:
        return None

    # Handle both videos and images
    if video_sources[video_id].endswith(('.jpg', '.png')):
        frame = cv2.imread(video_sources[video_id])
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, conf=0.25)
        yield process_frame(frame, results)
    else:
        cap = cv2.VideoCapture(video_sources[video_id])
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default to ~30 FPS if fps is unavailable

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 500))
            results = model.predict(frame, conf=0.25)
            yield process_frame(frame, results)

            # Calculate processing time and adjust frame delay
            elapsed_time = time.time() - start_time
            time.sleep(max(0, frame_delay - elapsed_time))

        cap.release()

def process_frame(frame, results):
    """Process the frame by drawing boxes and applying predictions."""
    detections = results[0].boxes.data.cpu().numpy()
    boxes, scores, person_count = [], [], 0

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cls_idx = int(cls)
        if cls_idx < len(class_list) and 'person' == class_list[cls_idx]:
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

    if len(boxes) > 0:
        indices = apply_nms(boxes, scores, iou_threshold=0.4)
        for i in indices:
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = scores[i]
            person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"person {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    cv2.putText(frame, f'Total Persons: {person_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    """Home page with video ID links."""
    video_links = "".join([f'<li><a href="/video?id={video_id}" class="video-link">Video {video_id}</a></li>'
                           for video_id in video_sources])
    return f"""
    <html><body><h1>Video Dashboard</h1><ul>{video_links}</ul></body></html>
    """

@app.route('/video')
def video_feed():
    """Video feed route."""
    video_id = request.args.get('id')
    return Response(generate_frames(video_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
