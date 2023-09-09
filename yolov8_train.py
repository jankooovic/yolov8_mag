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
flipud = image flip up-down (probability)
fliplr = image flip left-right (probability)
degrees = image rotation (+/- deg)
perspective = image perspective (+/- fraction), range 0-0.001
scale = image scale (+/- gain)
imgsz = 640	image size as scalar or (h, w) list, i.e. (640, 480)

### Sample commands:
model.train(data='config.yaml', epochs=100, imgsz=640)
model.train(data='config.yaml', epochs=300, imgsz=640, flipud=1, fliplr=1, degrees=90.0)

### Treniraj na določeni posamezni točki
model.train(data='config_aF1.yaml', epochs=300, imgsz=640, batch=1)
model.train(data='config_TKC.yaml', epochs=300, imgsz=640, batch=1)
model.train(data='config_TML.yaml', epochs=300, imgsz=640, batch=1)
model.train(data='config_FHC.yaml', epochs=300, imgsz=640, batch=1)
model.train(data='config_ALL.yaml', epochs=300, pretrained=True, imgsz=640, batch=1, degrees=10, perspective=0.0005)
"""

### Treniraj model - sprememba config datotek
model.train(
    data='config/config_ALL.yaml',
    epochs=300,
    pretrained=True,
    imgsz=2560,
    batch=1,
    degrees=10,
    perspective=0.0005,
    translate=0,
    scale=0.1,
    fliplr=0,
    mosaic=0)