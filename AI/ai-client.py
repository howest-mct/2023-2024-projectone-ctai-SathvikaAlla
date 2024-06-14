import socket
import threading
import cv2
from ultralytics import YOLO
import tempfile
import streamlit as st
import numpy as np
import os
# Server address for the Raspberry Pi
server_address = ('192.168.168.167', 8500)
client_socket = None
receive_thread = None
shutdown_flag = threading.Event()

# Define colors for bounding boxes
bbox_colors = {
    'Acne': (0, 255, 0),
    'Eczema': (255, 0, 0),
    'Chicken Skin' : (0, 0, 255)
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

def process_image(img, conf_threshold):
    global current_state, disease_detected
    results = model(img)
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
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, f"{label} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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

    # Display the annotated img, predictions, and confidences
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_placeholder.image(img, channels="RGB")
    results_placeholder.text("Detected Results:\n" + "\n".join(predictions))
    confidence_placeholder.text("Confidences:\n" + "\n".join(confidences))

type_detected = False
def process_img_skin_types(img, conf_threshold):
    global current_state, type_detected
    img = np.array(img)
    prediction = None
    confidence = None
    # Perform inference using the YOLOv8 skin type classification model
    results = model2.predict(img)

    # Get the predicted class and confidence
    for result in results:
        class_id = result.probs.top1
    # Check if the confidence is above the threshold
        if  float(result.probs.top1conf)>= conf_threshold:
            label = model2.names[int(class_id)]

            type_detected = True

            # Draw a bounding box around the image with the predicted class and confidence
            img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({float(result.probs.top1conf):.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if label == 'dry':
            if current_state != 'Dry':
                client_socket.sendall('dry'.encode('utf-8'))
                current_state = 'Dry'
            type_detected = True
        elif label == 'oily':
            if current_state != 'Oily':
                client_socket.sendall('oily'.encode('utf-8'))
                current_state = 'Oily'
            type_detected = True
        elif label == 'normal':
            if current_state != 'Normal':
                client_socket.sendall('normal'.encode('utf-8'))
                current_state = 'Normal'
            type_detected = True
    # Display the annotated image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_placeholder.image(img, channels="RGB")
    results_placeholder.text(f"Predicted Skin Type: {label} (Confidence: {float(result.probs.top1conf):.2f})")


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
                # decode is expecting bytes but we're giving it an array. CHANGE THIS

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
model2 = YOLO('AI\skin_types_best_1.pt')

# Streamlit page configuration
st.set_page_config(page_title="Skin Image Recognition")
st.title("Skin Image Recognition")



# Main model selection
st.subheader("Select the Model to Run 👉")
model_choice = st.selectbox(
        label = "Select a model to Run ",
        options = ("Skin Disease Detection", "Skin Type Classification"),
        index = None,
        label_visibility='collapsed'
    )


if model_choice == "Skin Disease Detection":
    # Skin disease detection page
    st.subheader("Skin Disease Detection")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Streamlit placeholders for image, results, and feeder status
    image_placeholder = st.empty()
    results_placeholder = st.empty()
    confidence_placeholder = st.empty()

    # Initial states for feeder and detection
    current_state = 'close'
    disease_detected = False

    # Webcam or upload video choice
    st.sidebar.title("Data Source")
    source = st.sidebar.radio("Choose the data source", ("Webcam", "Raspberry Pi Camera", "Upload Video", "Upload Image"), index=None)

    if source == "Raspberry Pi Camera":
        client_socket.sendall('start_video'.encode('utf-8'))
        process_video_from_socket(client_socket, conf_threshold)
    elif source == "Webcam":
        process_video(0, conf_threshold)
    elif source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            process_video(tmp_file_path, conf_threshold)
    elif source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["jpg","png","webp","jpeg"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            img = cv2.imread(tmp_file_path)
            process_image(img, conf_threshold)

elif model_choice == "Skin Type Classification":
    # Skin type classification page
    st.subheader("Skin Type Classification")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Streamlit placeholders for image, results, and feeder status
    image_placeholder = st.empty()
    results_placeholder = st.empty()
    confidence_placeholder = st.empty()

    # Initial states for feeder and detection
    current_state = 'close'
    skin_type_detected = False

    # Webcam or upload image choice
    st.sidebar.title("Data Source")
    source = st.sidebar.radio("Choose the data source", ("Upload Image","Webcam"), index=None)

    if source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg","png","webp","jpeg"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            img = cv2.imread(tmp_file_path)
            process_img_skin_types(img, conf_threshold)

    if source == 'Webcam':
        picture = st.camera_input("Take a picture")

        if picture is not None:
            # To read image file buffer with OpenCV:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            process_img_skin_types(cv2_img, conf_threshold)
