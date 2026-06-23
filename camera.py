import cv2
import torch
import numpy as np
from datetime import datetime
import time
import os
import glob

# Constants
MAX_SAVED_FILES = 50  # Maximum number of saved detection pairs (image + text)
SAVE_INTERVAL = 30    # Save every 30 seconds

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Colors for visualization
COLORS = np.random.uniform(0, 255, size=(80, 3))

# Initialize variables for FPS calculation
fps_start_time = 0
fps = 0

# Dictionary to store object counts
object_counts = {}

# Load YOLO model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

# Disable CUDA autocast warning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Function to save detection results
def save_detection(frame, detected_objects, faces):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"detections/detection_{timestamp}.jpg"
    text_path = f"detections/detection_{timestamp}.txt"
    
    # Save image
    cv2.imwrite(image_path, frame)
    
    # Save detection data
    with open(text_path, "w") as f:
        f.write(f"Time: {timestamp}\n")
        f.write(f"Objects detected: {', '.join(detected_objects)}\n")
        f.write(f"Faces detected: {len(faces)}\n")
        f.write(f"FPS: {fps:.2f}\n")

# Function to cleanup old detection files
def cleanup_old_files():
    # Get all jpg files (we'll use these as reference since each detection has both jpg and txt)
    jpg_files = glob.glob('detections/detection_*.jpg')
    jpg_files.sort()  # Sort by name (which includes timestamp)
    
    # If we have more files than our limit, remove the oldest ones
    if len(jpg_files) > MAX_SAVED_FILES:
        files_to_remove = jpg_files[:-MAX_SAVED_FILES]  # Keep the most recent MAX_SAVED_FILES
        for jpg_file in files_to_remove:
            txt_file = jpg_file.replace('.jpg', '.txt')
            try:
                os.remove(jpg_file)
                if os.path.exists(txt_file):
                    os.remove(txt_file)
            except Exception as e:
                print(f"Error removing file: {e}")

# Function to process video feed
def detect_objects_and_faces():
    global fps_start_time, fps
    fps_start_time = time.time()
    last_save_time = time.time()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    save_interval = 30  # Save every 30 seconds

    while True:
        current_time = time.time()
        
        # Calculate FPS
        if current_time - fps_start_time > 1:
            fps = frame_count / (current_time - fps_start_time)
            fps_start_time = current_time
            frame_count = 0

        ret, frame = cap.read()
        if not ret:
            break

        # Reset object counts for this frame
        object_counts.clear()

        # Perform object detection with YOLO
        results = yolo_model(frame)
        detected_objects = []

        # Extract detected objects with confidence > 0.5
        for *xyxy, conf, cls in results.xyxy[0]:  # xyxy format for better box drawing
            if conf > 0.5:  # confidence threshold
                label = results.names[int(cls)]
                detected_objects.append(f"{label} ({conf:.2f})")
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, xyxy[:4])
                color = COLORS[int(cls) % len(COLORS)].tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label_text = f"{label} {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update object counts
                object_counts[label] = object_counts.get(label, 0) + 1

        # Display FPS and timestamp
        fps_text = f"FPS: {fps:.2f}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 60
        for obj, count in object_counts.items():
            count_text = f"{obj}: {count}"
            cv2.putText(frame, count_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

        # Perform face detection with OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Face and Object Detection', frame)

        # Display detected objects
        if detected_objects:
            print("Detected objects:", ", ".join(detected_objects))

        # Save detection results periodically (every 30 seconds)
        if current_time - last_save_time >= SAVE_INTERVAL:
            save_detection(frame, detected_objects, faces)
            last_save_time = current_time
            cleanup_old_files()  # Cleanup old files after saving new ones
            cleanup_old_files()
        
        frame_count += 1

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects_and_faces()
