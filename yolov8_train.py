""" Train YOLOv8-pose model on the COCO128-pose dataset. """
from ultralytics import YOLO

"""
### Load a model - nano 3.3M params
model = YOLO('yolov8n-pose.yaml')   # build a new model from YAML
model = YOLO('yolov8n-pose.pt')  # load a pretrained model
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')   # build from YAML and transfer weights
"""
### Load a model - small 11.6M params
model = YOLO('yolov8s-pose.yaml')   # build a new model from YAML
model = YOLO('yolov8s-pose.pt')  # load a pretrained model
model = YOLO('yolov8s-pose.yaml').load('yolov8s-pose.pt')   # build from YAML and transfer weights

"""
# Re-Train the model:
hsv_h = 0.015 image HSV-Hue augmentation (fraction)
hsv_s = 0.7	image HSV-Saturation augmentation (fraction)
hsv_v = 0.4	image HSV-Value augmentation (fraction)

degrees = 0.0 image rotation (+/- deg)
scale = 0.5 image scale (+/- gain)
perspective = 0.0 image perspective (+/- fraction), range 0-0.001
fliplr = 0.5 image flip left-right (probability)
imgsz = 640	image size as scalar or (h, w) list, i.e. (640, 480)
batch = 16 number of images per batch (-1 for AutoBatch)
optimizer = Adam
lr0 = 0.01

Sample command: model.train(data='config.yaml', epochs=100, imgsz=640) 
"""

img_sizes = [1920 ,3680] #960, 1280, 1920, 2016, 3040, 3680
models = ["SGD"] # "SGD", "Adamax", "Adam"
config = 'config/config_ALL.yaml'

### Train model - per config file
for m in models:
        for img_size in img_sizes:
            model.train(
                data=config,
                imgsz=img_size,
                pretrained=True,
                epochs=300,
                patience=50,
                batch=16,
                lr0 = 0.01,
                optimizer=m,
                # Data augemntation parameters
                degrees=10,
                scale=0.1,
                perspective=0.001,
                # annoyance
                translate=0,
                fliplr=0,
                mosaic=0,
                hsv_h = 0,
                hsv_s = 0,
                hsv_v = 0,
                # multi GPUs
                device=[0,1],
                data_parallel=True
            )
