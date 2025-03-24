'''from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")'''
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pass any model type
results = model.train(epochs=5)

# Perform object detection on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Visualize the results
for result in results:
    result.show()