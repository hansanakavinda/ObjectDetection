# ObjectDetection GUI (YOLO)

Interactive desktop (Tk / CustomTkinter) demos for running Ultralytics YOLO models on images and live webcam video, evolving across versions from a minimal prototype to a richer UI that saves annotated outputs and lists class names with confidence scores.

## Highlights
- Image detection with bounding boxes (multiple variants).
- Live webcam detection (threaded and non‑threaded versions).
- Progressive UI improvements (layout, responsive resize, canvas scaling).
- Display of detected class names + confidence (v6).
- Saving annotated images and recorded live video feed (v7).
- Example YOLO training call (`app.py`) for quick experimentation.

## Repository Scripts Overview
| Script | Purpose |
|--------|---------|
| `main.py`, `centered.py`, `object_detection_app.py` | Early single‑image detection GUIs. |
| `fixed_image_resize.py`, `auto_resize_image.py` | Experiments with dynamic image resizing. |
| `video_object_detection.py`, `video_object_detection_v2.py` | Initial live webcam detection attempts. |
| `VOD_v3_threadding.py` | Introduces threading for smoother live updates. |
| `VOD_v4_YOLOv8.py` | Switches to Ultralytics `yolov8n.pt` weight file. |
| `VOD_v5_UI_improvements.py` | Responsive canvas + better resizing behavior. |
| `VOD_v6_namesAndConfidentScore.py` | Adds listing of detected object names + confidence %. |
| `VOD_v7_SavingResults.py` | Adds saving annotated images and recording live feed videos. (Recommended start) |
| `utils.py` | Helper for loading and resizing images for display. |
| `app.py` | Example: brief YOLO training then inference on sample image. |
| `newApp.py` | Unrelated Tk demo (unit conversion) kept for reference. |

## Recommended Entry Point
Start with the most feature‑complete version:

```cmd
python VOD_v7_SavingResults.py
```

Then: select an image OR start live detection. Annotated images go to `saved_images/`, recorded webcam sessions go to `saved_live_feed/` as timestamped MP4 files.

## Model Weights
Included weight files:
- `yolo11n.pt` (likely a newer/custom YOLO nano model – ensure compatibility with your Ultralytics install).
- `yolov8n.pt` (official Ultralytics YOLOv8 nano model).

Switching models: edit any script line like:
```python
model = YOLO("yolov8n.pt")
```
You can download other sizes (e.g. `yolov8s.pt`, `yolov8m.pt`) via:
```cmd
pip install ultralytics
```
and the model will auto‑download on first use.

## Dependencies
Detected from imports and `requirements.txt` :
- ultralytics
- opencv-python (cv2)
- numpy
- pillow (PIL)
- customtkinter (USED but missing from list)
- tkinter (bundled with standard Python on Windows; not installed via pip)
- threading / time (standard library)

### Installation (Windows CMD)
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy pillow customtkinter
```
OR if you fix the file name:
```cmd
pip install -r requirements.txt
```

## Usage Examples
Single image detection (simple UI):
```cmd
python main.py
```

Live detection with threading + confidence listing:
```cmd
python VOD_v6_namesAndConfidentScore.py
```

Latest version with saving outputs:
```cmd
python VOD_v7_SavingResults.py
```

YOLO training sample (quick 5 epoch run):
```cmd
python app.py
```