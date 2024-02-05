""" Validate YOLOv8 model. """
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')   # build a new model from YAML
model = YOLO('./runs/pose/train_SGD_noparams/weights/best.pt')

# Validate the model
# check config file and chnage path to MAC
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category