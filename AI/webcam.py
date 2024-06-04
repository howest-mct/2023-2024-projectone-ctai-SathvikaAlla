import cv2
from ultralytics import YOLO

# Specify the correct path to your trained model weights
model_path = r"AI\skin_disease_best_4.pt"  # Correct path based on your directory structure

# Load the trained model
model = YOLO(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main loop
try:
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Could not read frame.")
            break

        # Make predictions
        results = model.predict(source=frame, save=False, save_txt=False, conf=0.4)

        # Render results on the frame
        result_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('Webcam', result_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()