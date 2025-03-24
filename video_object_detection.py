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

# Configure grid weights for full responsiveness
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Main frame (centered container)
mainframe = ctk.CTkFrame(root)
mainframe.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

# Ensure elements are centered
mainframe.grid_columnconfigure(0, weight=1)
mainframe.grid_rowconfigure(1, weight=1)

# Global variable to hold the current image for resizing
current_img = None
video_capture = None
video_streaming = False

# Function to process the image from file upload
def detect_objects():
    global current_img  # Use the global variable to store the image
    
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

    # Update the image size based on the window size
    update_image_size(img)

# Function to update the image size when the window is resized
def update_image_size(img=None):
    # If no image is provided (during window resizing), use the current image
    if not img:
        img = current_img
    
    if not img:
        return

    # Get the current window dimensions
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Calculate the new image size (keeping the aspect ratio)
    new_width = window_width - 40  # Padding of 20 on both sides
    new_height = int((new_width / img.width) * img.height) if img else window_height - 100  # Keep aspect ratio
    
    # Resize image
    img = img.resize((new_width, new_height))

    # Convert to CustomTkinter-compatible image
    ctk_image = ctk.CTkImage(light_image=img, size=(new_width, new_height))

    # Update label with resized image
    img_label.configure(image=ctk_image)
    img_label.image = ctk_image

# Function to start live webcam object detection
def start_live_detection():
    global video_capture, video_streaming

    # Open the webcam (camera index 0 for default)
    video_capture = cv2.VideoCapture(0)
    video_streaming = True
    update_live_frame()

# Function to stop live webcam detection
def stop_live_detection():
    global video_capture, video_streaming
    video_streaming = False
    if video_capture:
        video_capture.release()

# Function to process webcam frames
def update_live_frame():
    global video_capture, video_streaming

    if not video_streaming or not video_capture.isOpened():
        return
    
    # Read frame from webcam
    ret, frame = video_capture.read()
    
    if not ret:
        return

    # Run YOLO object detection on the frame
    results = model(frame)
    img_with_boxes = results[0].plot()  # Get image with bounding boxes

    # Convert frame to PIL format
    img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Update the live frame display
    update_image_size(img)

    if video_streaming:
        # Schedule the next frame update
        root.after(10, update_live_frame)

# Button to select image
btn_select_image = ctk.CTkButton(mainframe, text="Select Image", command=detect_objects)
btn_select_image.grid(row=0, column=0, pady=10, padx=10, sticky="n")

# Button to start live webcam detection
btn_start_live = ctk.CTkButton(mainframe, text="Start Live Detection", command=start_live_detection)
btn_start_live.grid(row=1, column=0, pady=10, padx=10, sticky="n")

# Button to stop live webcam detection
btn_stop_live = ctk.CTkButton(mainframe, text="Stop Live Detection", command=stop_live_detection)
btn_stop_live.grid(row=2, column=0, pady=10, padx=10, sticky="n")

# Image display label (centered)
img_label = ctk.CTkLabel(mainframe, text="", anchor="center")
img_label.grid(row=3, column=0, pady=10, padx=10, sticky="n")

# Run Tkinter event loop
root.mainloop()
