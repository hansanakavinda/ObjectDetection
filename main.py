from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import utils

import customtkinter as ctk

# Initialize the application with a modern theme
ctk.set_appearance_mode("System")  # Options: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Options: "blue" (default), "green", "dark-blue"

# Initialize the main window
root = ctk.CTk()
root.title("Object Detection App")
root.geometry("800x600")


# Load the YOLO model
model = YOLO("yolo11n.pt")

# Initialize Tkinter window
'''root = tk.Tk()
root.title("Object Detection App")
root.geometry("600x500") 
'''
mainframe = ctk.CTkFrame(root)
mainframe.grid(row=0, column=0, sticky="nsew")
mainframe.grid(padx=10, pady=10)


# Configure grid weights for responsiveness
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)


'''mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1) 
'''
# Function to process the image
def detect_objects():
    global img_label
    
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return

    # Run YOLO object detection
    results = model(file_path)
    img_with_boxes = results[0].plot()  # Get image with bounding boxes

    # Convert image to PIL format for Tkinter
    img = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((400, 300))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)

    # Update label with detected image
    img_label.configure(image=img_tk)
    img_label.image = img_tk

# Create UI elements
'''btn = tk.Button(mainframe, text="Select Image", command=detect_objects, font=("Arial", 14))
btn.pack(pady=10)'''

btn = ctk.CTkButton(mainframe, text="Select Image", command=detect_objects)
btn.grid(row=0, column=0, pady=10, padx=10)


'''img_label = tk.Label(mainframe)  # Label to display the image
img_label.pack()'''

img_label = ctk.CTkLabel(mainframe)
img_label.grid(row=1, column=0, pady=10, padx=10)

# Run Tkinter event loop
root.mainloop()