import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Initialize the application with a modern theme
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")  

# Initialize the main window
root = ctk.CTk()
root.title("Object Detection App")
root.geometry("900x600")
root.minsize(800, 500)  # Minimum window size

# Configure grid layout for responsiveness
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Main frame (container)
mainframe = ctk.CTkFrame(root)
mainframe.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
mainframe.grid_rowconfigure(0, weight=1)
mainframe.grid_columnconfigure(1, weight=1)  # Right frame takes remaining space

# Left frame (Fixed size, buttons)
left_frame = ctk.CTkFrame(mainframe, width=200)
left_frame.grid(row=0, column=0, sticky="nsw", padx=10)
left_frame.grid_propagate(False)  # Prevents frame from resizing when buttons are clicked

# Right frame (Dynamic resizing)
right_frame = ctk.CTkFrame(mainframe)
right_frame.grid(row=0, column=1, sticky="nsew", padx=10)
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

# Canvas for displaying images & video (allows dynamic resizing)
canvas = ctk.CTkCanvas(right_frame, bg="black")
canvas.grid(row=0, column=0, sticky="nsew")

# Global variables
current_img = None
video_capture = None
video_streaming = False
video_thread = None  

# Function to process image upload
def detect_objects():
    global current_img  
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    results = model(file_path)  # Run YOLO detection
    img_with_boxes = results[0].plot()  

    img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    current_img = img  
    update_display(img)

# Function to update the displayed image/video dynamically
def update_display(img):
    if not img:
        return

    # Get the canvas size
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Maintain aspect ratio
    img_ratio = img.width / img.height
    canvas_ratio = canvas_width / canvas_height

    if img_ratio > canvas_ratio:
        new_width = canvas_width
        new_height = int(canvas_width / img_ratio)
    else:
        new_height = canvas_height
        new_width = int(canvas_height * img_ratio)

    img = img.resize((new_width, new_height))

    # Convert to a Tkinter-compatible image
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_img, anchor="center")
    canvas.image = tk_img  

# Function to start live webcam detection (runs in a separate thread)
def start_live_detection():
    global video_capture, video_streaming, video_thread

    if video_streaming:  
        return

    video_capture = cv2.VideoCapture(0)
    video_streaming = True

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

        results = model(frame)
        img_with_boxes = results[0].plot()  

        img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        root.after(0, update_display, img)

# Button to select image
btn_select_image = ctk.CTkButton(left_frame, text="Select Image", command=detect_objects)
btn_select_image.pack(pady=10, padx=10, fill="x")

# Button to start live webcam detection
btn_start_live = ctk.CTkButton(left_frame, text="Start Live Detection", command=start_live_detection)
btn_start_live.pack(pady=10, padx=10, fill="x")

# Button to stop live webcam detection
btn_stop_live = ctk.CTkButton(left_frame, text="Stop Live Detection", command=stop_live_detection)
btn_stop_live.pack(pady=10, padx=10, fill="x")

# Bind window resize event to dynamically update images/videos
root.bind("<Configure>", lambda event: update_display(current_img))

# Run Tkinter event loop
root.mainloop()
