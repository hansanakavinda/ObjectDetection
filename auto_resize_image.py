import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

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

# Function to process the image
def detect_objects():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    # Run YOLO object detection
    results = model(file_path)
    img_with_boxes = results[0].plot()  # Get image with bounding boxes

    # Convert image to PIL format
    img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Update the image size based on the window size
    update_image_size(img)

# Function to update the image size when the window is resized
def update_image_size(img=None):
    # Get the current window dimensions
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Calculate the new image size (keeping the aspect ratio)
    new_width = window_width - 150  # Padding of 20 on both sides
    new_height = int((new_width / img.width) * img.height) if img else window_height - 100  # Keep aspect ratio
    
    # Resize image
    img = img.resize((new_width, new_height))
    
    # Convert to CustomTkinter-compatible image
    ctk_image = ctk.CTkImage(light_image=img, size=(new_width, new_height))

    # Update label with resized image
    img_label.configure(image=ctk_image)
    img_label.image = ctk_image

# Button to select image
btn = ctk.CTkButton(mainframe, text="Select Image", command=detect_objects)
btn.grid(row=0, column=0, pady=10, padx=10, sticky="n")

# Image display label (centered)
img_label = ctk.CTkLabel(mainframe, text="", anchor="center")
img_label.grid(row=1, column=0, pady=10, padx=10, sticky="n")

# Bind window resizing to update image size
root.bind("<Configure>", lambda event: update_image_size())

# Run Tkinter event loop
root.mainloop()
