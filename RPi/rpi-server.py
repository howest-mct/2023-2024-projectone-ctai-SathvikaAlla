import socket
import threading
import time
from RPi import GPIO
import cv2
import numpy as np
from lcd import LCD
import time
# GPIO setup for the servo motor
SERVO_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# Global vars
client_socket = None
server_socket = None
server_thread = None
shutdown_flag = threading.Event()
lcd = LCD()
# Function to set the servo motor angle
def set_servo_angle(angle):
    duty = angle / 18 + 3
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(1)  # Allow the servo to move
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)  # Stop the servo
    print(f"Servo moved to {angle} degrees")

# Function to setup the server socket and start listening for connections
def setup_socket_server():
    global server_socket, server_thread, shutdown_flag
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8500))  # Bind to all interfaces on port 8500
    server_socket.settimeout(0.2)  # Set a timeout for socket operations
    server_socket.listen(1)  # Listen for incoming connections

    # Start a thread to accept incoming connections
    server_thread = threading.Thread(target=accept_connections, args=(shutdown_flag,), daemon=True)
    server_thread.start()

# Function to accept incoming client connections
def accept_connections(shutdown_flag):
    global client_socket
    print("Accepting connections")
    while not shutdown_flag.is_set():
        try:
            client_socket, addr = server_socket.accept()
            print("Connected by", addr)
            client_thread = threading.Thread(target=handle_client, args=(client_socket, shutdown_flag,))
            client_thread.start()
        except socket.timeout:
            pass

# Function to handle communication with the client
def handle_client(sock, shutdown_flag):
    global cap, lcd
    try:

        while not shutdown_flag.is_set():
            data = sock.recv(512)  # Receive data from the client
            if not data:
                break
            message = data.decode(errors='ignore')
            print(f"Received from client: {message}")
            if message == 'start_video':
                cap = cv2.VideoCapture(0)  # Open the Raspberry Pi camera
                if not cap.isOpened():
                    print("Failed to open camera")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                send_video(sock)

            elif message == 'acne':
                lcd.clear()
                lcd.send_string("Acne detected",1)
            elif message == 'eczema':
                lcd.clear()
                lcd.send_string("Eczema detected",1)
            elif message == 'chicken_skin':
                lcd.clear()
                lcd.send_string("Chiken Skin detected",1)
            elif message in ['no_disease', 'source_change', 'model_change']:
                lcd.clear()

            elif message == 'dry':
                set_servo_angle(20)
                time.sleep(1)
                set_servo_angle(0)
            elif message == 'normal':
                set_servo_angle(90)
                time.sleep(1)
                set_servo_angle(0)
            elif message == 'oily':
                set_servo_angle(160)
                time.sleep(1)
                set_servo_angle(0)
            elif message == 'no_type':
                set_servo_angle(0)

    except socket.timeout:
        pass
    except Exception as e:
        print(f"Error in handle_client: {e}")
    finally:
        if 'cap' in globals():
            cap.release()
        sock.close()

# Function to send video frames to the client
def send_video(sock):
    global cap
    try:
        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                print("Failed to read from camera")
                break
            img_encode = cv2.imencode('.jpg', frame)[1]

            # Converting the image into numpy array
            data_encode = np.asarray(img_encode)

            # Converting the array to bytes.
            byte_encode = data_encode.tobytes()
            sock.sendall((byte_encode + b"END_FRAME"))  # Add a delimiter to separate frames
    except Exception as e:
        print(f"Error in send_video: {e}")
    finally:
        if 'cap' in globals():
            cap.release()
            print("Camera released")

# Main program loop
try:
    setup_socket_server()
    set_servo_angle(0)

    while True:
        time.sleep(10)  # Keep the main thread alive
except KeyboardInterrupt:
    print("Server shutting down")
    shutdown_flag.set()
    set_servo_angle(0)
finally:
    server_thread.join()
    server_socket.close()
    lcd.clear()
    servo.stop()
    GPIO.cleanup()