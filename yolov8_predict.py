""" Use a trained YOLOv8n-pose model to run predictions on images. """
#https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Keypoints

from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions
import re

# Dataset path:
path = "./data/dataset/"
save_path = "./data/predicted"
test_img_path = "/images/test/"
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'ALL', 'sTMA', 'sFDMA']
landmark_names = ['sTMA1', 'sTMA2', 'FHC', 'sFMDA1', 'sFMDA2','TKC', 'TML', 'FNOC', 'aF1'] # based on labels in config file
imgsize = 3680 # check if the same as trained model
model_paths = {"ALL" : "./runs/pose/train_ALL_" + str(imgsize) + "_grayscale/weights/best.pt"}
model_paths = {"ALL" : "./runs/pose/train_SGD_"+ imgsize + "_small_batch8/weights/best.pt"}
skipped = []


# create dataset archive
yolov8_functions.dataset_archive(save_path)

directories = yolov8_functions.get_dirs(path)

# script
for directory in directories:
    print("Directory path:", directory + test_img_path)

    # select correct point name based on directory
    point_name = next((name for name in point_names if name in directory), None)

    if point_name is None:
        continue

    image_paths = yolov8_functions.get_jpg_paths("./" + directory + test_img_path)

    # select correct model based on point
    for img_path in image_paths:
        skip = False
        model_path = model_paths.get(point_name, None)
    
        if model_path is None:
            continue

        # load correct model of yolov8
        yolov8_model = YOLO(model_path)  # load a custom model

        # Run inference on image with arguments - same imgsize as training
        results = yolov8_model.predict(img_path,imgsz=imgsize)  # predict on an image 

        name = yolov8_functions.filename_creation(img_path, ".jpg")
        filename = save_path + "/" + name

        # read image
        img = cv2.imread(img_path)
        temp = np.array(img)
        img_shape = temp.shape
        img_shape = [img_shape[1], img_shape[0]]
        
        # Save JSON file with data
        dictionary = {
            "Image name": name,
            "Image_size": img_shape,
        }
        landmarks = [] # landmarks list
        labels = [] # labels list
        skipped_points = []
        skipped_labels = []
        removed_points = []
        removed_labels = []
        i = 0
        conf_box = []
        conf_pose = []

        for result in results:
            for idx, keypoint in enumerate(result.keypoints):
                point = keypoint.xy.tolist()

                x = point[0][0][0]
                y = point[0][0][1]
                landmark = [x,y]

                # get label abd point names from result
                label = int(result.boxes.cls[idx])
                label = landmark_names[label]

                if landmark[0] < img_shape[0]*0.15 or landmark[0] > img_shape[0]*0.85:
                    skipped_points.append(landmark)
                    skipped_labels.append(label)
                    continue

                labels.append(label)
                landmarks.append(landmark)
                conf_box.append(float(result.boxes.conf[idx]))
                conf_pose.append(float(result.keypoints.conf[idx]))
            
            # check landmarks for duplicates
            if len(labels) > 9:

                duplicates = {x for x in labels if labels.count(x) > 1}
                duplicates = list(duplicates)

                # get array indexes for duplicates
                remove_occurances = []
                for dup in duplicates:
                    occurances = yolov8_functions.get_indices(dup, labels)

                    # get confidence for each duplicate
                    confs = []
                    for idx in occurances:
                        conf = conf_box[idx] + conf_pose[idx]
                        confs.append(conf)
                    
                    # get highest confidence of all duplicates or first highest confidence - upgrade to average
                    highest_conf_idx = confs.index(max(confs))
                    del occurances[highest_conf_idx]

                    for occ in occurances:
                        remove_occurances.append(occ)

                # remove other occurances from labels, landmarks, etc
                remove_occurances.sort(reverse=True)
                for idx in remove_occurances:
                    del labels[idx]
                    del landmarks[idx]
                    del conf_box[idx]
                    del conf_pose[idx]

            for idx, landmark in enumerate(landmarks):
                dictionary.update({
                    labels[idx]:landmark,
                })

            dictionary.update({
                    "Skipped points":skipped_points,
                    "Skipped labels":skipped_labels,
                    "Removed points":removed_points,
                    "removed labels":removed_labels
                })

        yolov8_functions.save_prediction_image(landmarks, temp, filename)
        yolov8_functions.create_json_datafile(dictionary, filename)