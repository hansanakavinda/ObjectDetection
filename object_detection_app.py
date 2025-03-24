import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
from ultralytics import YOLO

# Initialize CustomTkinter appearance
ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

# Create the main application window
root = ctk.CTk()
root.title("Object Detection App")
root.geometry("800x600")

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Create the main frame
mainframe = ctk.CTkFrame(root)
mainframe.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Configure grid weights for responsiveness
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

def detect_objects():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Run YOLO object detection
    results = model(file_path)
    img_with_boxes = results[0].plot()  # Get image with bounding boxes

    # Convert image to PIL format
    img = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

    # Create a CTkImage object
    img_ctk = ctk.CTkImage(light_image=img, size=(400, 300))

    # Update label with the CTkImage
    img_label.configure(image=img_ctk)
    img_label.image = img_ctk

# Create UI elements
btn = ctk.CTkButton(mainframe, text="Select Image", command=detect_objects)
btn.grid(row=0, column=0, pady=10, padx=10)

img_label = ctk.CTkLabel(mainframe, text="")
img_label.grid(row=1, column=0, pady=10, padx=10)

# Run the application
root.mainloop()
