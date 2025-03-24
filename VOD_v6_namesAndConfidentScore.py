import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Initialize UI
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Object Detection App")
root.geometry("900x600")
root.minsize(800, 500)

# Configure layout
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Load YOLO Model
model = YOLO("yolo11n.pt")

# Frames
mainframe = ctk.CTkFrame(root)
mainframe.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
mainframe.grid_rowconfigure(0, weight=1)
mainframe.grid_columnconfigure(1, weight=1)

left_frame = ctk.CTkFrame(mainframe, width=200)
left_frame.grid(row=0, column=0, sticky="nsw", padx=10)
left_frame.grid_propagate(False)

right_frame = ctk.CTkFrame(mainframe)
right_frame.grid(row=0, column=1, sticky="nsew", padx=10)
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

# Canvas for images & video
canvas = ctk.CTkCanvas(right_frame, bg="black")
canvas.grid(row=0, column=0, sticky="nsew")

# Label for detected objects (Multiline)
results_label = ctk.CTkLabel(
    right_frame, text="", font=("Arial", 14), wraplength=600, justify="left"
)
results_label.grid(row=1, column=0, sticky="ew", pady=5)

# Global Variables
current_img = None
video_capture = None
video_streaming = False
video_thread = None  
mode = "image"  # Track if we are in "image" or "live" mode

# Function to process image upload
def detect_objects():
    global current_img, mode
    
    mode = "image"  # Switch to image mode
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    results = model(file_path)  
    img_with_boxes = results[0].plot()  

    img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    current_img = img  
    update_display(img)

    # Extract detected objects
    detected_text = extract_detected_objects(results)
    results_label.configure(text=detected_text)

# Function to extract object names & confidence scores
def extract_detected_objects(results):
    detected_items = []
    for r in results:
        frame_objects = []
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]  
            conf = float(box.conf[0]) * 100  
            frame_objects.append(f"{cls} ({conf:.2f}%)")
        
        if frame_objects:
            detected_items.append(", ".join(frame_objects))  # Show objects per frame in a single line
    
    return "\n".join(detected_items) if detected_items else "No objects detected."

# Function to update displayed image/video
def update_display(img):
    if mode == "live":
        return  # Prevent image updates when live detection is running

    if not img:
        return

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_ratio = img.width / img.height
    canvas_ratio = canvas_width / canvas_height

    if img_ratio > canvas_ratio:
        new_width = canvas_width
        new_height = int(canvas_width / img_ratio)
    else:
        new_height = canvas_height
        new_width = int(canvas_height * img_ratio)

    img = img.resize((new_width, new_height))
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_img, anchor="center")
    canvas.image = tk_img  

# Start live webcam detection (threaded)
def start_live_detection():
    global video_capture, video_streaming, video_thread, mode

    if video_streaming:  
        return

    mode = "live"  # Switch to live mode
    video_capture = cv2.VideoCapture(0)
    video_streaming = True

    video_thread = threading.Thread(target=update_live_frame, daemon=True)
    video_thread.start()

# Stop live webcam detection
def stop_live_detection():
    global video_capture, video_streaming, mode
    video_streaming = False
    mode = "image"  # Switch back to image mode

    if video_capture:
        video_capture.release()

# Process webcam frames (threaded)
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

        detected_text = extract_detected_objects(results)
        
        root.after(0, lambda: update_display_live(img))  
        root.after(0, lambda: results_label.configure(text=detected_text))  

# Update the display for live video
def update_display_live(img):
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_ratio = img.width / img.height
    canvas_ratio = canvas_width / canvas_height

    if img_ratio > canvas_ratio:
        new_width = canvas_width
        new_height = int(canvas_width / img_ratio)
    else:
        new_height = canvas_height
        new_width = int(canvas_height * img_ratio)

    img = img.resize((new_width, new_height))
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_img, anchor="center")
    canvas.image = tk_img  

# Buttons
btn_select_image = ctk.CTkButton(left_frame, text="Select Image", command=detect_objects)
btn_select_image.pack(pady=10, padx=10, fill="x")

btn_start_live = ctk.CTkButton(left_frame, text="Start Live Detection", command=start_live_detection)
btn_start_live.pack(pady=10, padx=10, fill="x")

btn_stop_live = ctk.CTkButton(left_frame, text="Stop Live Detection", command=stop_live_detection)
btn_stop_live.pack(pady=10, padx=10, fill="x")

# Bind window resize
root.bind("<Configure>", lambda event: update_display(current_img))

# Run UI loop
root.mainloop()
