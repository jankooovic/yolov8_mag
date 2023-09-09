from ultralytics import YOLO

"""
Validate trained YOLOv8n-pose model accuracy on the COCO128-pose dataset. 
No argument need to passed as the model retains it's training data and arguments as model attributes.
"""

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO("/opt/homebrew/runs/pose/train32_imgsize_1280/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

"""
ToDo:
- Naredi, da se validira model po treniranju

"""