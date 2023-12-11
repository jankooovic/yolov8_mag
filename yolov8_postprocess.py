""" Postprocess predictions """
from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions
import json
import matplotlib.pyplot as plt

# Dataset path:
predicted_path = "./data/predicted/"
images_path = "./data/dataset/ALL/images/test/"
landmark_names = ['sTMA1', 'sTMA2', 'FHC', 'sFMDA1', 'sFMDA2','TKC', 'TML', 'FNOC', 'aF1']
skipped_path = "data/predicted/skipped.json"

# create dataset archive
#yolov8_functions.dataset_archive(save_path)

# get image
image_paths = yolov8_functions.get_dirs(images_path)

# get skipped images
to_skip = []
with open(predicted_path + "skipped.json") as f:
        data = json.load(f)
        to_skip = (data['Skipped images'])

# remove skipped images
for skip in to_skip:
    image_paths.remove(skip)

#print(image_paths)
#print(to_skip)

# Get points
with open(predicted_path + "skipped.json") as f:
        data = json.load(f)
        to_skip = (data['Skipped images'])

json_paths_predicted = [directory for directory in yolov8_functions.get_dirs(predicted_path) if ".json" in str(directory)]
json_paths_predicted.remove(skipped_path)
#print(json_paths_predicted)

# sort paths:
image_paths = sorted(image_paths)
json_paths_predicted = sorted(json_paths_predicted)


for idx, img_path in enumerate(image_paths):
    
    # load points
    predicted_coordinates = []
    with open(json_paths_predicted[idx]) as f:
        data = json.load(f)
        for name in landmark_names:
            predicted_coordinates.append(data[name])

    #print("Image path:", img_path, "Point:", json_paths_predicted[idx])
    #print("Predicted coordinates:", predicted_coordinates)

    # load image
    image = cv2.imread(img_path)

    # perform the canny edge detector to detect image edges
    #edges = cv2.Canny(image, threshold1=60, threshold2=150)
    #edges = cv2.Canny(image, threshold1=100, threshold2=200, apertureSize = 5, L2gradient = True)

    plt.imshow(image, cmap="gray")
    plt.show()




# Find edges for FNOC, sTMA, sFDMA, 

# Find midlle of edged for TML

# Allign TKC, FHC better? 

# Algorithm for aF1 - FNOC & FHC coordinates

# write new points to ladnamrks file

