from ultralytics import YOLO

"""
Train a YOLOv8-pose model on the COCO128-pose dataset.
"""

### Load a model
# build a new model from YAML
model = YOLO('yolov8n-pose.yaml')  
# load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.pt')
# build from YAML and transfer weights
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt') 

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

### Sample commands:
model.train(data='config.yaml', epochs=100, imgsz=640)
model.train(data='config.yaml', epochs=300, imgsz=640, flipud=1, fliplr=1, degrees=90.0)

"""
"""
### Treniraj model - Cela slika
model.train(
    data='config/config_ALL.yaml',
    epochs=300,
    pretrained=True,
    imgsz=960,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)
"""

### Treniraj model - FHC
model.train(
    data='config/config_FHC.yaml',
    epochs=300,
    pretrained=True,
    imgsz=960,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)

### Treniraj model - aF1
model.train(
    data='config/config_aF1.yaml',
    epochs=300,
    pretrained=True,
    imgsz=960,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)

### Treniraj model - TML
model.train(
    data='config/config_TML.yaml',
    epochs=300,
    pretrained=True,
    imgsz=960,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)

### Treniraj model - TKC
model.train(
    data='config/config_TKC.yaml',
    epochs=300,
    pretrained=True,
    imgsz=960,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)

### Treniraj model - FNOC
model.train(
    data='config/config_FNOC.yaml',
    epochs=300,
    pretrained=True,
    imgsz=960,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)