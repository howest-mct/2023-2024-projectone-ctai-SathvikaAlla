import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# Specify the correct path to your trained model weights
model_path = r"AI\skin_disease_best_4.pt"  # Correct path based on your directory structure

# Load the trained model
model = YOLO(model_path)

# Load your object detection model
def detect_objects(frame):
    global model
    # Process the frame with your object detection model
    results = model.predict(source=frame, save=False, save_txt=False, conf=0.4)
    result_frame = results[0].plot()

    return result_frame

# Function to capture video from webcam
def capture_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected_frame = detect_objects(frame)
        yield detected_frame[:, :, ::-1]  # Gradio expects RGB, but OpenCV provides BGR

# Create a Gradio interface to display the video stream
iface = gr.Interface(
    fn=capture_video,
    inputs=[],
    outputs="webcam"
)

# Launch the interface
iface.launch()
