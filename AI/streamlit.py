# import streamlit as st
# import cv2
# from ultralytics import YOLO
# import numpy as np
# import tempfile

# # Load the model
# model = YOLO("AI/skin_disease_best_4.pt")

# # Page configuration
# st.set_page_config(page_title="Skin Disease Recognition", layout="wide")
# st.title("Skin Image Recognition")

# # Threshold slider
# conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# image_placeholder = st.empty()
# results_placeholder = st.empty()

# # Define class IDs for helmet, vest, and shoes
# ACNE_CLASS_ID = 0
# ECZEMA_CLASS_ID = 1
# CHICKEN_SKIN_CLASS_ID = 2

# # Process video frames
# def process_video(video_path, conf_threshold, frame_skip=10):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         # Inference
#         results = model(frame)
#         annotated_frame = frame.copy()
#         predictions = []
#         detected_items = set()

#         # Extract results
#         for result in results:
#             for bbox in result.boxes.data:
#                 x1, y1, x2, y2, score, class_id = bbox
#                 if score >= conf_threshold:
#                     label = model.names[int(class_id)]
#                     predictions.append(f"Class: {label}, Confidence: {score:.2f}")
#                     cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                     cv2.putText(annotated_frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     detected_items.add(int(class_id))



#         missing_text = '<span style="color:green">All items detected</span>'

#         predictions.append(missing_text)

#         annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         image_placeholder.image(annotated_frame, channels="RGB")

#         results_placeholder.markdown("<br>".join(predictions), unsafe_allow_html=True)

#     # Release the video capture
#     cap.release()

# # Webcam or upload video choice
# st.sidebar.title("Video Source")
# source = st.sidebar.radio("Choose the video source", ("Webcam", "Upload"))

# if source == "Webcam":
#     # Initialize video capture
#     cap = cv2.VideoCapture(0)

#     process_video(0, conf_threshold)

# else:
#     uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_file_path = tmp_file.name


#         process_video(tmp_file_path, conf_threshold)

from ultralytics import YOLO

model = YOLO("AI\skin_types_best.pt")
model.predict("AI\images.jpg")