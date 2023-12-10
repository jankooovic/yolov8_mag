""" Postprocess predictions """
from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions

# Dataset path:
predicted_path = "./data/predicted"
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'ALL', 'sTMA', 'sFDMA']
landmark_names = ['sTMA1', 'sTMA2', 'FHC', 'sFMDA1', 'sFMDA2','TKC', 'TML', 'FNOC', 'aF1']


# create dataset archive
#yolov8_functions.dataset_archive(save_path)


# Get points

# check image size

# remove all points that are inside 20% of the picture width from edge