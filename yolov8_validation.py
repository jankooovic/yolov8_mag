""" Validate YOLOv8-pose model on the COCO128-pose dataset. """
from ultralytics import YOLO

imgsize = 1920
model_path = "./runs/pose/train_ALL_" + str(imgsize) + "_grayscale/weights/best.pt"

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO(model_path)  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category