from flask import Flask, request, render_template, Response
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Specify the correct path to your trained model weights
model_path = r"AI\skin_disease_best_4.pt"  # Correct path based on your directory structure

# Load the trained model
model = YOLO(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

# Function to generate frames from the webcam
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Make predictions
            results = model.predict(source=frame, save=False, save_txt=False, conf=0.4)
            # Render results on the frame
            result_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Root URL route that renders the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file found", 400
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(source=img, save=False, save_txt=False, conf=0.4)
    result_img = results[0].plot()
    _, img_encoded = cv2.imencode('.jpg', result_img)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

# Route to handle video uploads
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file found", 400
    file = request.files['video']
    file.save('uploaded_video.mp4')
    cap = cv2.VideoCapture('uploaded_video.mp4')
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, save=False, save_txt=False, conf=0.4)
        result_frame = results[0].plot()
        frames.append(result_frame)
    cap.release()
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for frame in frames:
        out.write(frame)
    out.release()
    return "Video processed and saved as output_video.mp4"

# Route to stream video feed from webcam
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
