""" Validate YOLOv8 model. """
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')   # build a new model from YAML
m = './eksperimenti_26122023/eksperiment2 - izbira velikosti vhodne slike/'
w = '/weights/best.pt'

model = YOLO(m + 'train_adam_3680_params' + w) # path to custom model

# Validate the model
# check config file and chnage path to MAC
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each categorys