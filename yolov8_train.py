""" Train YOLOv8-pose model on the COCO128-pose dataset. """
from ultralytics import YOLO

### Load a model
model = YOLO('yolov8n-pose.yaml')   # build a new model from YAML
model = YOLO('yolov8n-pose.pt')  # load a pretrained model
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')   # build from YAML and transfer weights

"""
# Re-Train the model:
hsv_h = 0.015 image HSV-Hue augmentation (fraction)
hsv_s = 0.7	image HSV-Saturation augmentation (fraction)
hsv_v = 0.4	image HSV-Value augmentation (fraction)
degrees = 0.0 image rotation (+/- deg)
scale = 0.5 image scale (+/- gain)
shear = 0.0	image shear (+/- deg)
perspective = 0.0 image perspective (+/- fraction), range 0-0.001
flipud = 0.0 image flip up-down (probability)
fliplr = 0.5 image flip left-right (probability)
imgsz = 640	image size as scalar or (h, w) list, i.e. (640, 480)
batch = 16 number of images per batch (-1 for AutoBatch)

Sample command: model.train(data='config.yaml', epochs=100, imgsz=640) 
"""

### Treniraj model - per config file
#config_files = ['config/config_ALL.yaml', 'config/config_FHC.yaml', 'config/config_aF1.yaml', 'config/config_FNOC.yaml', 'config/config_TKC.yaml', 'config/config_TML.yaml']
config_files = ['config/config_ALL.yaml']

for config in config_files:

    print("Using config file:", config)

    model.train(
        data=config,
        pretrained=True,
        epochs=300,
        imgsz=960,
        #batch=-1,
        hsv_h = 0.015,
        hsv_s = 0.1,
        hsv_v = 0.05,
        degrees=10,
        scale=0.1,
        perspective=0.0005,
        translate=0,
        fliplr=0,
        mosaic=0
        )