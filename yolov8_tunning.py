""" Tune YOLOv8-pose model on the COCO128-pose dataset. """
from ultralytics import YOLO

config_file = 'config/config_ALL.yaml'

### Load a model
model = YOLO('yolov8n-pose.yaml')   # build a new model from YAML
model = YOLO('yolov8n-pose.pt')  # load a pretrained model
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')   # build from YAML and transfer weights


# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(data='coco8.config_file', use_ray=True)