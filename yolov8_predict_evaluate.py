
""" Use a trained YOLOv8n-pose model to run predictions on images. """
from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions

# Dataset path:
path = "/Users/jankovic/Downloads/yolov8/data/dataset/"
sav_path = "/Users/jankovic/Downloads/yolov8/data/predicted/"

"""
# yolov8 custom_model_path
model_path_all = "/opt/homebrew/runs/pose/train_all_imgsize_640/weights/best.pt"
model_path_tml = "/opt/homebrew/runs/pose/train_all_imgsize_640/weights/best.pt"
model_path_tkc = "/opt/homebrew/runs/pose/train_all_imgsize_640/weights/best.pt"
model_path_af1 = "/opt/homebrew/runs/pose/train_all_imgsize_640/weights/best.pt"
model_path_fhc = "/opt/homebrew/runs/pose/train_all_imgsize_640/weights/best.pt"
"""
################################ - premakni v direktorij yolov8/models - ko imam boljše rezultate
model_path_all = "/opt/homebrew/runs/pose/train_all_imgsize_960/weights/best.pt"
model_path_fhc = "/opt/homebrew/runs/pose/train_fhc_imgsize_960/weights/best.pt"
model_path_af1 = "/opt/homebrew/runs/pose/train_af1_imgsize_960/weights/best.pt"
model_path_fnoc = "/opt/homebrew/runs/pose/train_fnoc_imgsize_960/weights/best.pt"
model_path_tkc = "/opt/homebrew/runs/pose/train_tkc_imgsize_960/weights/best.pt"
model_path_tml = "/opt/homebrew/runs/pose/train_tml_imgsize_960/weights/best.pt"

###############################
model_path_train7 = "/opt/homebrew/runs/pose/train7/weights/best.pt"
model_path_test = "/opt/homebrew/runs/pose/train7/weights/best.pt"

# Test image directories
all_imgs = "/images/test"

# point names array
directories = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'ALL']

# Find all images&paths to images in test directories
## poti do direktorijev, kjer so .jpg datoteke
dirs = yolov8_functions.get_dirs(path)

# Create test images directory list
## poti do vseh test jpg slik
image_paths = yolov8_functions.get_jpg_paths(dirs, directories, path, all_imgs)

# izberi slike iz posameznega direktorija - for loop, da gre čez vse pathe
all_paths = yolov8_functions.get_paths_word("ALL", image_paths)
fhc_paths = yolov8_functions.get_paths_word("FHC", image_paths)
af1_paths = yolov8_functions.get_paths_word("aF1", image_paths)
fnoc_paths = yolov8_functions.get_paths_word("FNOC", image_paths)
tkc_paths = yolov8_functions.get_paths_word("TKC", image_paths)
tml_paths = yolov8_functions.get_paths_word("TML", image_paths)

# predict points na slikah posameznega direktorija -> ALL
for img_path in fhc_paths:

    # Load a model
    model = YOLO('yolov8n-pose.pt')  # load an official model
    model = YOLO(model_path_test)  # load a custom model

    # Predict with the model
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
        "Point names": directories,
        "Point coordinates": p,
        "Image_size": img_shape,
    }

    yolov8_functions.create_json_datafile(dictionary, filename)



# 1. predict
# 2. cascade - optional
# 3. compare and report results