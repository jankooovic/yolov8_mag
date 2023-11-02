
""" Use a trained YOLOv8n-pose model to run predictions on images. """
from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions

# Dataset path:
path = "./data/dataset/"
save_path = "./data/predicted"
test_img_path = "/images/test/"
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'ALL']
landmark_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1']

model_paths = [
    "./runs/pose/trained_ALL_960_intel/weights/best.pt",
    "./runs/pose/trained_FHC_960_intel/weights/best.pt",
    "./runs/pose/trained_aF1_960_intel/weights/best.pt",
    "./runs/pose/trained_FNOC_960_intel/weights/best.pt",
    "./runs/pose/trained_TKC_960_intel/weights/best.pt",
    "./runs/pose/trained_TML_960_intel/weights/best.pt"
]

# create dataset archive
yolov8_functions.dataset_archive(save_path)

directories = yolov8_functions.get_dirs(path)

# script
for directory in directories:
    print("Directory path:", directory + test_img_path)

    # select correct point name based on directory
    point_name = ""
    skipLoop = True
    for name in point_names:
        if name in directory:
            point_name = name
            skipLoop = False
    
    if skipLoop:
        continue

    image_paths = yolov8_functions.get_jpg_paths("./" + directory + test_img_path)

    # select correct model based on point
    for img_path in image_paths:

        model = ""
        skipLoop2 = True
        for model_path in model_paths:
            if point_name in model_path:
                model = model_path
                skipLoop2 = False
    
        if skipLoop2:
            continue

        # load correct model of yolov8
        yolov8_model = YOLO(model)  # load a custom model

        # Run inference on image with arguments - same imgsize as training
        results = yolov8_model.predict(img_path,imgsz=960)  # predict on an image 
        
        landmarks = [] # landmarks list
        for result in results:

            for keypoint_indx, keypoint in enumerate(result.keypoints):
                point = keypoint.xy.tolist()
                x = point[0][0][0]
                y = point[0][0][1]
                landmark = [x,y]
                landmarks.append(landmark)

        name = yolov8_functions.filename_creation(img_path, ".jpg")
        filename = save_path + "/" + name

        # read image
        img = cv2.imread(img_path)
        temp = np.array(img)
        img_shape = temp.shape
        yolov8_functions.save_prediction_image(landmarks, temp, filename)

        # Save JSON file with data
        dictionary = {
            "Image name": filename,
            "Point names": landmark_names,
            "Point coordinates": landmarks,
            "Image_size": img_shape,
        }

        yolov8_functions.create_json_datafile(dictionary, filename)
