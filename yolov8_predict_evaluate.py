
""" Use a trained YOLOv8n-pose model to run predictions on images. """
from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions

path = "./data/dataset/"
sav_path = "./data/predicted/"
test_img_path = "images/test/"
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'ALL']

model_paths = [
    "./data/runs/pose/ALL_imgsize_960/weights/best.pt",
    "./data/runs/pose/FHC_imgsize_960/weights/best.pt",
    "./data/runs/pose/aF1_imgsize_960/weights/best.pt",
    "./data/runs/pose/FNOC_imgsize_960/weights/best.pt",
    "./data/runs/pose/TKC_imgsize_960/weights/best.pt",
    "./data/runs/pose/TML_imgsize_960/weights/best.pt"
]

directories = yolov8_functions.get_dirs(path)

# script
for directory in directories:
    skipLoop = True
    print("Directory path:", directory + "/" + test_img_path)

    # select correct point name based on directory
    point_name = ""
    for name in point_names:
        if name in directory:
            point_name = name
            skipLoop = False
    
    if skipLoop:
        continue

    image_paths = yolov8_functions.get_jpg_paths("./" + directory + "/" + test_img_path)

    # select correct model based on point
    for img_path in image_paths:
        model = ""
        for model_path in model_paths:
            if point_name in model_path:
                model = model_path

        # load correct model of yolov8
        #model = YOLO('yolov8n-pose.pt')  # load an official model
        model = YOLO(model)  # load a custom model

        print("Model loaded:", model)

"""
        results = model(img_path)[0]  # predict on an image
        
        p = [] # landmarks list
        for result in results:
            #print(result.keypoints)

            for keypoint_indx, keypoint in enumerate(result.keypoints):
                point = keypoint.xy.tolist()
                x = point[0][0][0]
                y = point[0][0][1]
                landmark = [x,y]
                p.append(landmark)
                #print("X: ", x, "Y: ", y)

        #print("Landmarks: ",p)

        # read image
        img = cv2.imread(img_path)

        # Show and save image&points
        temp = np.array(img)
        img_shape = temp.shape
        name = yolov8_functions.filename_creation(path, img_path, ".jpg")
        name = name.replace("imagestest", "")
        filename = sav_path + name
        print("Name: ",name)

        yolov8_functions.full_image_save_predict(p, temp, filename)

        # Save JSON file with data
        # kreiraj JSON dataset
        dictionary = {
            "Image name": filename,
            "Point names": point_names,
            "Point coordinates": p,
            "Image_size": img_shape,
        }

        yolov8_functions.create_json_datafile(dictionary, filename)
"""

# 1. predict
# 2. cascade - optional
# 3. compare and report results