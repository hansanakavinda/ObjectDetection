import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Initialize the application with a modern theme
ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

# Initialize the main window
root = ctk.CTk()
root.title("Object Detection App")
root.geometry("800x600")

# Configure grid weights for responsiveness
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Main frame (centered container)
mainframe = ctk.CTkFrame(root)
mainframe.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

# Left frame for buttons
left_frame = ctk.CTkFrame(mainframe)
left_frame.grid(row=0, column=0, sticky="ns", padx=10)

# Right frame for image/video feed
right_frame = ctk.CTkFrame(mainframe)
right_frame.grid(row=0, column=1, sticky="nsew", padx=10)

# Ensure elements are centered
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

# Global variables
current_img = None
video_capture = None
video_streaming = False
video_thread = None  # Thread for video processing

# Function to process the image from file upload
def detect_objects():
    global current_img  
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    # Run YOLO object detection
    results = model(file_path)
    img_with_boxes = results[0].plot()  # Get image with bounding boxes

    # Convert image to PIL format
    img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    current_img = img  # Store the image globally

    # Update the image size
    update_image_size(img)

# Function to update the image size when window is resized
def update_image_size(img=None):
    if not img:
        img = current_img
    
    if not img:
        return

    # Get the current window dimensions
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Calculate new image size while maintaining aspect ratio
    new_width = window_width // 2 - 40  # Padding of 20 on both sides of the right frame
    new_height = int((new_width / img.width) * img.height) if img else window_height - 100  
    
    # Resize image
    img = img.resize((new_width, new_height))

    # Convert to CustomTkinter-compatible image
    ctk_image = ctk.CTkImage(light_image=img, size=(new_width, new_height))

    # Update label with resized image
    img_label.configure(image=ctk_image)
    img_label.image = ctk_image

# Function to start live webcam object detection (runs in a separate thread)
def start_live_detection():
    global video_capture, video_streaming, video_thread

    if video_streaming:  # Avoid starting multiple threads
        return

    # Open the webcam
    video_capture = cv2.VideoCapture(0)
    video_streaming = True

    # Start a new thread for video streaming
    video_thread = threading.Thread(target=update_live_frame, daemon=True)
    video_thread.start()

# Function to stop live webcam detection
def stop_live_detection():
    global video_capture, video_streaming
    video_streaming = False

    if video_capture:
        video_capture.release()

# Function to process webcam frames (runs in a separate thread)
def update_live_frame():
    global video_capture, video_streaming

    while video_streaming and video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run YOLO object detection on the frame
        results = model(frame)
        img_with_boxes = results[0].plot()  # Get image with bounding boxes

        # Convert frame to PIL format
        img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Update the live frame display
        root.after(0, update_image_size, img)

# Button to select image
btn_select_image = ctk.CTkButton(left_frame, text="Select Image", command=detect_objects)
btn_select_image.grid(row=0, column=0, pady=10, padx=10, sticky="n")

# Button to start live webcam detection
btn_start_live = ctk.CTkButton(left_frame, text="Start Live Detection", command=start_live_detection)
btn_start_live.grid(row=1, column=0, pady=10, padx=10, sticky="n")

# Button to stop live webcam detection
btn_stop_live = ctk.CTkButton(left_frame, text="Stop Live Detection", command=stop_live_detection)
btn_stop_live.grid(row=2, column=0, pady=10, padx=10, sticky="n")

# Image display label (centered)
img_label = ctk.CTkLabel(right_frame, text="", anchor="center")
img_label.grid(row=0, column=0, pady=10, padx=10, sticky="n")

# Run Tkinter event loop
root.mainloop()
