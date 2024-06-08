import socket
import threading
import cv2
from ultralytics import YOLO
import tempfile
import streamlit as st
import numpy as np

# Server address for the Raspberry Pi
server_address = ('192.168.168.167', 8500)
client_socket = None
receive_thread = None
shutdown_flag = threading.Event()

# Define colors for bounding boxes
bbox_colors = {
    'Acne': (0, 255, 0),  # Green for Orange
    'Eczema': (255, 0, 0),  # Red for Niuniu
    'Chicken Skin' : (0,0,255)
}

# Function to setup the client socket and connect to the server
def setup_socket_client():
    global client_socket, receive_thread
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    print("Connected to server")

    # Start a thread to receive messages from the server
    receive_thread = threading.Thread(target=receive_messages, args=(client_socket, shutdown_flag))
    receive_thread.start()

# Function to receive messages from the server
def receive_messages(sock, shutdown_flag):
    sock.settimeout(1)
    try:
        while not shutdown_flag.is_set():
            try:
                data = sock.recv(1024)
                if not data:
                    break
                print("Received from server:", data.decode(errors='ignore'))
                # Send an acknowledgment back to the server
            except socket.timeout:
                continue
    except Exception as e:
        if not shutdown_flag.is_set():
            print(f"Connection error: {e}")
    finally:
        sock.close()


# Function to process video from a local camera or file
def process_video(video_source, conf_threshold, frame_skip=5):
    global current_state, disease_detected

    # Open the video source (webcam or file)
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size for the video capture
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video source
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to reduce processing load

        # Perform inference using the YOLO model
        results = model(frame)
        annotated_frame = frame.copy()
        predictions = []
        confidences = []

        # Check if any cat is detected with confidence above threshold
        disease_detected = False
        for result in results:
            for bbox in result.boxes.data:
                x1, y1, x2, y2, score, class_id = bbox
                if score >= conf_threshold:
                    label = model.names[int(class_id)]
                    predictions.append(f"Class: {label}")
                    confidences.append(f"Confidence: {score:.2f}")
                    # Draw bounding box and label on the frame with specified colors
                    color = bbox_colors.get(label, (255, 255, 255))  # Default to white if label not found
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(annotated_frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Send commands to the feeder based on detection results
                    if label == 'Acne':
                        if current_state != 'Acne':
                            client_socket.sendall('acne'.encode('utf-8'))
                            current_state = 'Acne'
                        disease_detected = True
                    elif label == 'Eczema':
                        if current_state != 'Eczema':
                            client_socket.sendall('eczema'.encode('utf-8'))
                            current_state = 'Eczema'
                        disease_detected = True
                    elif label == 'Chicken Skin':
                        if current_state != 'Chicken Skin':
                            client_socket.sendall('chicken_skin'.encode('utf-8'))
                            current_state = 'Chicken Skin'
                        disease_detected = True



        # Display the annotated frame, predictions, and confidences
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(annotated_frame, channels="RGB")
        results_placeholder.text("Detected Results:\n" + "\n".join(predictions))
        confidence_placeholder.text("Confidences:\n" + "\n".join(confidences))

    cap.release()

# Function to process video received from the Raspberry Pi over a socket
def process_video_from_socket(sock, conf_threshold, frame_skip=5):
    global current_state, disease_detected

    frame_count = 0
    data = b""

    while True:
        try:
            packet = sock.recv(4096)  # Receive data from the socket
            if not packet:
                break
            data += packet

            # Assuming that each frame is separated by a known delimiter
            frames = data.split(b"END_FRAME")

            for frame in frames[:-1]:
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue  # Skip frames to reduce processing load

                # Decode frame from the received data
                np_arr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    print("Failed to decode frame")
                    continue

                # Perform inference using the YOLO model
                results = model(img)
                annotated_frame = img.copy()
                predictions = []
                confidences = []

                # Check if any cat is detected with confidence above threshold
                disease_detected = False
                for result in results:
                    for bbox in result.boxes.data:
                        x1, y1, x2, y2, score, class_id = bbox
                        if score >= conf_threshold:
                            label = model.names[int(class_id)]
                            predictions.append(f"Class: {label}")
                            confidences.append(f"Confidence: {score:.2f}")
                            # Draw bounding box and label on the frame with specified colors
                            color = bbox_colors.get(label, (255, 255, 255))  # Default to white if label not found
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(annotated_frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                            # Send commands to the feeder based on detection results
                            if label == 'Acne':
                                if current_state != 'Acne':
                                    client_socket.sendall('acne'.encode('utf-8'))
                                    current_state = 'Acne'
                                disease_detected = True
                            elif label == 'Eczema':
                                if current_state != 'Eczema':
                                    client_socket.sendall('eczema'.encode('utf-8'))
                                    current_state = 'Eczema'
                                disease_detected = True
                            elif label == 'Chicken Skin':
                                if current_state != 'Chicken Skin':
                                    client_socket.sendall('chicken_skin'.encode('utf-8'))
                                    current_state = 'Chicken Skin'
                                disease_detected = True

                # Display the annotated frame, predictions, and confidences
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(annotated_frame, channels="RGB")
                results_placeholder.text("Detected Results:\n" + "\n".join(predictions))
                confidence_placeholder.text("Confidences:\n" + "\n".join(confidences))

            data = frames[-1]  # Keep the last incomplete frame
        except Exception as e:
            print(f"Error receiving video frame: {e}")
            break


# Initialize the client socket connection
setup_socket_client()

# Load the YOLO model
model = YOLO('AI\skin_disease_best_4.pt')

# Streamlit page configuration
st.set_page_config(page_title="Skin Disease Recognition", layout="wide")
st.title("Skin Disease Recognition")

# Slider to set the confidence threshold
conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Streamlit placeholders for image, results, and feeder status
image_placeholder = st.empty()
results_placeholder = st.empty()
confidence_placeholder = st.empty()

# Initial states for feeder and detection
current_state = 'close'
disease_detected = False

# Webcam or upload video choice
st.sidebar.title("Video Source")
source = st.sidebar.radio("Choose the video source", ("Raspberry Pi Camera", "Webcam", "Upload"))

if source == "Raspberry Pi Camera":
    client_socket.sendall('start_video'.encode('utf-8'))
    process_video_from_socket(client_socket, conf_threshold)
elif source == "Webcam":
    process_video(0, conf_threshold)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        process_video(tmp_file_path, conf_threshold)