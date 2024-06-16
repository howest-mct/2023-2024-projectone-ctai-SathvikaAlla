import socket
import threading
import cv2
from ultralytics import YOLO
import tempfile
import streamlit as st
import numpy as np
import os
import time
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
        if not disease_detected:
            if current_state != 'Nothing':
                client_socket.sendall('no_disease'.encode('utf-8'))
                current_state = 'Nothing'


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

    if not disease_detected:
        if current_state != 'Nothing':
            client_socket.sendall('no_disease'.encode('utf-8'))
            current_state = 'Nothing'
    # Display the annotated img, predictions, and confidences
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_placeholder.image(img, channels="RGB")
    results_placeholder.text("Detected Results:\n" + "\n".join(predictions))
    confidence_placeholder.text("Confidences:\n" + "\n".join(confidences))

def process_img_skin_types(img, conf_threshold):
    global current_state_type, type_detected
    img = np.array(img)
    prediction = None
    confidence = None
    # Perform inference using the YOLOv8 skin type classification model
    results = model2.predict(img)
    type_detected = False

    # Get the predicted class and confidence
    for result in results:
        class_id = result.probs.top1
    # Check if the confidence is above the threshold
        if  float(result.probs.top1conf)>= conf_threshold:
            label = model2.names[int(class_id)]

            type_detected = True

            # Draw a bounding box around the image with the predicted class and confidence
            img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 2)

        if label == 'dry':
            if current_state_type != 'Dry':
                client_socket.sendall('dry'.encode('utf-8'))
                current_state_type = 'Dry'
            type_detected = True
        elif label == 'oily':
            if current_state_type != 'Oily':
                client_socket.sendall('oily'.encode('utf-8'))
                current_state_type = 'Oily'
            type_detected = True
        elif label == 'normal':
            if current_state_type != 'Normal':
                client_socket.sendall('normal'.encode('utf-8'))
                current_state_type = 'Normal'
            type_detected = True

    if not type_detected:
            if current_state_type != 'Nothing':
                client_socket.sendall('no_type'.encode('utf-8'))
                current_state_type = 'Nothing'
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
st.set_page_config(page_title="Skin Image Recognition", layout='wide')
st.title("Skin Image Recognition")

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# Main model selection
st.subheader("Select the Model to Run 👉")
model_choice = st.selectbox(
        label = "Select a model to Run ",
        options = ("Skin Disease Detection", "Skin Type Classification"),
        index = None,
        label_visibility='collapsed'
    )


if not model_choice:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Choose the detected skin type")
        type_choice = st.selectbox(
            label = "Select a model to Run ",
            options = ("Dry", "Normal", "Oily"),
            index = None,
            label_visibility='collapsed'
        )

    with col2:
        st.subheader("Choose the detected skin disease")
        disease_choice = st.selectbox(
            label = "Select a model to Run ",
            options = ("Acne", "Eczema", "Chicken Skin", "None"),
            index = None,
            label_visibility='collapsed'
        )
    eczema_text = """
    # Tips for Managing Eczema

    ## Skincare Routine

    1. **Gentle Cleansing**:
        Use a mild, fragrance-free cleanser.

    2. **Moisturize**:
        Apply a thick, fragrance-free moisturizer immediately after bathing.

    3. **Bathing**:
        Take lukewarm baths and limit to 10-15 minutes.
        Add colloidal oatmeal or baking soda to soothe the skin.

    4. **Topical Treatments**:
        Use over-the-counter hydrocortisone cream or prescribed topical steroids for flare-ups.

    5. **Avoid Irritants**:
        Use fragrance-free and hypoallergenic skincare and laundry products.

    ## Lifestyle Tips

    1. **Hydration**:
        Drink plenty of water daily.

    2. **Humidifier**:
        Use a humidifier to maintain moisture in the air.

    3. **Wear Soft Fabrics**:
        Choose clothing made of cotton or other soft materials.

    4. **Avoid Scratching**:
        Keep nails short and wear gloves at night if needed.

    5. **Stress Management**:
        Practice stress-reducing activities like yoga or meditation.

    ## Professional Help

    1. **Consult a Dermatologist**:
        Seek advice for stronger treatments or if over-the-counter options are ineffective.

    2. **Allergy Testing**:
        Consider testing for potential allergens that may trigger eczema.
    """

    chicken_skin_text = """
    # Tips for Managing Chicken Skin (Keratosis Pilaris)

    ## Possible Causes

    1. **Keratin Buildup**:
        Excess keratin blocks hair follicles, forming rough, small bumps.

    2. **Genetics**:
        Often runs in families and is linked to certain genetic traits.

    3. **Dry Skin**:
        More common in individuals with dry skin.

    4. **Other Skin Conditions**:
        Associated with eczema and other dry skin conditions.

    ## Tips for Managing Chicken Skin

    ### Skincare Routine

    1. **Gentle Cleansing**:
        Use a mild, fragrance-free cleanser.

    2. **Exfoliate**:
        Apply a gentle exfoliant with alpha-hydroxy acids (AHA) or beta-hydroxy acids (BHA) to remove dead skin cells.

    3. **Moisturize**:
        Use a thick, fragrance-free moisturizer, preferably with urea or lactic acid.

    4. **Short, Warm Showers**:
        Limit shower time and use lukewarm water to avoid drying out the skin.

    5. **Avoid Harsh Products**:
        Stay away from soaps and lotions with harsh chemicals and fragrances.

    ### Lifestyle Tips

    1. **Humidifier**:
        Use a humidifier to add moisture to the air, especially in dry climates or during winter.

    2. **Wear Soft Fabrics**:
        Opt for clothing made from soft, breathable fabrics like cotton.

    3. **Stay Hydrated**:
        Drink plenty of water to keep skin hydrated from within.

    4. **Avoid Picking**:
        Refrain from picking or scratching the bumps to prevent irritation and infection.

    ### Professional Help

    1. **Consult a Dermatologist**:
        Seek professional advice for stronger treatments if over-the-counter products are ineffective.

    2. **Prescription Treatments**:
        Consider prescription creams with retinoids or stronger exfoliants.

    3. **Laser Therapy**:
        Explore laser treatments to reduce redness and smooth skin texture.

    Consistency is key. Regular application of skincare products and lifestyle adjustments can significantly improve the appearance of chicken skin.
    """

    acne_oily_text = """
    # Tips for Clearing Acne with an Oily Skin Type

    ## Skincare Routine

    1. **Cleanse Twice Daily**:
        Use a gentle, foaming, non-comedogenic cleanser for oily or acne-prone skin.

    2. **Exfoliate Regularly**:
        Apply a chemical exfoliant with salicylic or glycolic acid 1-2 times a week.

    3. **Toner**:
        Choose an alcohol-free toner with witch hazel, salicylic acid, or tea tree oil.

    4. **Treatment Products**:
        Use benzoyl peroxide, retinoids, or niacinamide to reduce acne and inflammation.

    5. **Moisturize**:
        Opt for an oil-free, non-comedogenic moisturizer.

    6. **Sunscreen**:
        Apply a broad-spectrum, oil-free SPF 30+ daily.

    ## Lifestyle Tips

    1. **Diet**:
        Limit high-glycemic foods and eat antioxidant-rich foods.

    2. **Hydration**:
        Drink plenty of water daily.

    3. **Avoid Touching Your Face**:
        Prevent oil, dirt, and bacteria transfer.

    4. **Hair Care**:
        Keep hair clean and away from your face.

    5. **Regular Exercise**:
        Exercise regularly and cleanse your face afterward.

    6. **Stress Management**:
        Engage in stress-reducing activities like yoga or meditation.

    ## Professional Treatments

    1. **Consult a Dermatologist**:
        Seek advice for stronger treatments if OTC products fail.

    2. **Professional Procedures**:
        Consider chemical peels, laser therapy, or extractions for persistent acne.

    Consistency is essential. If over-the-counter treatments are not effective, consult a dermatologist.
    """

    acne_normal_text = """
    # Tips for Managing Acne with Normal Skin

    ## Skincare Routine

    1. **Cleanse**:
        Use a gentle, non-comedogenic cleanser twice daily.

    2. **Exfoliate**:
        Use salicylic or glycolic acid 2-3 times a week.

    3. **Moisturize**:
        Apply a lightweight, non-comedogenic moisturizer.

    4. **Treat**:
        Use benzoyl peroxide or retinoids for acne.

    5. **Sunscreen**:
        Use oil-free, SPF 30+ daily.

    ## Lifestyle Tips

    1. **Diet**:
        Eat antioxidant-rich foods.

    2. **Hydrate**:
        Drink plenty of water.

    3. **Avoid Touching Face**:
        Keep hands off your face.

    4. **Exercise**:
        Exercise regularly and cleanse after.

    5. **Manage Stress**:
        Practice yoga or meditation.

    ## Professional Help

    1. **Dermatologist**:
        Consult for stronger treatments if needed.

    2. **Procedures**:
        Consider chemical peels or laser therapy.
    """

    acne_dry_text = """
    # Tips for Managing Acne with Dry Skin

    ## Skincare Routine

    1. **Gentle Cleansing**:
        Use a mild, hydrating cleanser twice daily.

    2. **Moisturize**:
        Apply a rich, non-comedogenic moisturizer to maintain hydration.

    3. **Exfoliate**:
        Use a gentle chemical exfoliant with lactic acid (AHA) once a week.

    4. **Treatment Products**:
        Use benzoyl peroxide or salicylic acid sparingly to avoid over-drying.
        Consider using a hyaluronic acid serum to boost moisture.

    5. **Sunscreen**:
        Apply a broad-spectrum SPF 30+ that is hydrating and non-comedogenic daily.

    ## Lifestyle Tips

    1. **Hydration**:
        Drink plenty of water throughout the day.

    2. **Avoid Harsh Products**:
        Steer clear of alcohol-based products and strong astringents.

    3. **Humidifier**:
        Use a humidifier to add moisture to the air, especially in dry climates or winter.

    4. **Balanced Diet**:
        Consume foods rich in omega-3 fatty acids and antioxidants.

    5. **Gentle Handling**:
        Avoid hot water and abrasive scrubbing; pat skin dry instead of rubbing.

    ## Professional Treatments

    1. **Consult a Dermatologist**:
        Seek advice for tailored treatments if OTC products are insufficient.

    2. **Professional Procedures**:
        Consider treatments like hydrating facials or mild chemical peels designed for dry skin.

    """

    if type_choice == 'Oily' and disease_choice == 'Acne':
        st.write_stream(stream_data(acne_oily_text))
    elif type_choice == 'Dry' and disease_choice == 'Acne':
        st.write_stream(stream_data(acne_dry_text))
    elif type_choice == 'Normal' and disease_choice=='Acne':
        st.write_stream(stream_data(acne_normal_text))


    elif type_choice=='Oily' and disease_choice == 'Eczema':
        st.write_stream(stream_data(eczema_text))
    elif type_choice=='Normal' and disease_choice == 'Eczema':
        st.write_stream(stream_data(eczema_text))
    elif type_choice == 'Dry' and disease_choice == 'Eczema':
        st.write_stream(stream_data(eczema_text))

    elif type_choice=='Oily' and disease_choice == 'Chicken Skin':
        st.write_stream(stream_data(chicken_skin_text))
    elif  type_choice=='Normal' and disease_choice == 'Chicken Skin':
        st.write_stream(stream_data(chicken_skin_text))
    elif type_choice == 'Dry' and disease_choice == 'Chicken Skin':
        st.write_stream(stream_data(chicken_skin_text))


if model_choice == "Skin Disease Detection":
    # Skin disease detection page
    st.subheader("Skin Disease Detection")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Streamlit placeholders for image, results, and feeder status
    image_placeholder = st.empty()
    results_placeholder = st.empty()
    confidence_placeholder = st.empty()
    resources_diseases = st.empty()


    # Initial states for feeder and detection
    current_state = ''
    disease_detected = False

    # Webcam or upload video choice
    st.sidebar.title("Data Source")
    source = st.sidebar.radio("Choose the data source", ("Live Webcam", "Image from Webcam", "Upload Video", "Upload Image"), index=None)


    if source == "Live Webcam":
        process_video(0, conf_threshold)

    elif source == 'Image from Webcam':
        picture = st.camera_input("Take a picture")

        if picture is not None:
            # To read image file buffer with OpenCV:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            process_image(cv2_img, conf_threshold)


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

    current_state_type = ''
    skin_type_detected = False

    # Webcam or upload image choice
    st.sidebar.title("Data Source")
    source = st.sidebar.radio("Choose the data source", ("Upload Image","Image from Webcam"), index=None)

    if source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg","png","webp","jpeg"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            img = cv2.imread(tmp_file_path)
            process_img_skin_types(img, conf_threshold)

    if source == 'Image from Webcam':
        picture = st.camera_input("Take a picture")

        if picture is not None:
            # To read image file buffer with OpenCV:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            process_img_skin_types(cv2_img, conf_threshold)
