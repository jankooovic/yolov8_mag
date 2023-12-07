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

optimizer = SGD, Adam, Adamax
lr0 = 0.01 - 0.0001

Sample command: model.train(data='config.yaml', epochs=100, imgsz=640) 
"""

learning_rates = [0.01] #0.1, 0.001, 0.0001
optimizers = ["Adam", "SGD", "Adamax"]
img_sizes = [960, 1920] #960, 1280, 1920, 2016, 3040, 3680
config_files = ['config/config_ALL.yaml']

### Train model - per config file
for config in config_files:
    for imgsize in img_sizes:
        for optimizer in optimizers:
            for lr in learning_rates:

                print("Using config file:", config)
                print("using optimizer:", optimizer)
                print("Using learning rate:", lr)

                model.train(
                    data=config,
                    pretrained=True,
                    epochs=300,
                    batch=16,
                    # Test parameters
                    imgsz=imgsize,
                    lr0 = lr,
                    optimizer=optimizer,
                    # Data augemntation parameters
                    hsv_h = 0.015,
                    hsv_s = 0.1,
                    hsv_v = 0.05,
                    degrees=10,
                    scale=0.1,
                    perspective=0.001,
                    translate=0,
                    fliplr=0,
                    mosaic=0
                    )