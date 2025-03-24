import cv2
from PIL import Image, ImageTk

def load_image_for_display(image_path, width=400, height=300):
    """ Load an image and resize it for display in Tkinter """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((width, height))
    return ImageTk.PhotoImage(img)
