""" Use a trained YOLOv8n-pose model to run predictions on images. """
#https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Keypoints

from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions
import re
from sklearn.decomposition import PCA
import json
from scipy.spatial.distance import euclidean

# Dataset path:
dataset_path = "./data/dataset/"
save_path = "./data/predicted"
test_img_path = "/images/test/"
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'ALL', 'sTMA', 'sFDMA']
landmark_names = ['sTMA1', 'sTMA2', 'FHC', 'sFMDA1', 'sFMDA2','TKC', 'TML', 'FNOC', 'aF1'] # based on labels in config file
imgsize = 1920 # check if the same as trained model
#model_paths = {"ALL" : "./runs/pose/train_ALL_" + str(imgsize) + "_grayscale/weights/best.pt"}
model_paths = {"ALL" : "./runs/pose/train_SGD_"+ str(imgsize) + "_small_batch8/weights/best.pt"}
skipped = []

# PCA script - to check and remove duplicates
predicted_path = "./data/predicted/"
test_path = "./data/dataset/JSON/"
test_images_path =  "./data/dataset/ALL/images/test/"
postprocess_path = "./data/postprocess/"
skipped_path = 'data/postprocess/skipped.json'
#save_path = "./data/postprocess"
images_path = "./data/dataset/ALL/images/test/"
landmark_names_pca = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
point_names_all = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA', 'sTMA', 'TML']
false_prediction = []
image_name = None

# Test points PCA
aF1_points_t = []
fhc_points_t = []
fnoc_points_t = []
tkc_points_t = []
sfdma1_points_t = []
sfdma2_points_t = []
stma1_points_t = []
stma2_points_t = []
tml_points_t = []

# create dataset archive
yolov8_functions.dataset_archive(save_path)


### PCA learning on test points ###
# get only paths that are to be evaluated from test
json_paths_test = [path for path in yolov8_functions.get_dirs(test_path) if not any(name in path for name in point_names_all)]
json_paths_test = sorted(json_paths_test)

for idx, path in enumerate(json_paths_test):
    skip = False

    # Test points json
    p_names = []
    img_size = []
    test_coordinates = []
    with open(path) as f:
        data = json.load(f)
        for coord in point_names_all:
            if coord == 'sTMA':
                stma1_x = data[coord][0]
                stma1_y = data[coord][1]
                stma2_x = data[coord][2]
                stma2_y = data[coord][3]
                stma1 = [stma1_x, stma1_y]
                stma2 = [stma2_x, stma2_y]
                test_coordinates.append(stma1)
                p_names.append('sTMA1')
                test_coordinates.append(stma2)
                p_names.append('sTMA2')
            elif coord == 'sFMDA':
                stma1_x = data[coord][0]
                stma1_y = data[coord][1]
                stma2_x = data[coord][2]
                stma2_y = data[coord][3]
                stma1 = [stma1_x, stma1_y]
                stma2 = [stma2_x, stma2_y]
                test_coordinates.append(stma1)
                p_names.append('sFMDA1')
                test_coordinates.append(stma2)
                p_names.append('sFMDA2')
            else:
                test_coordinates.append(data[coord])
                p_names.append(coord)
        img_size =  data['Image_size']  # x,y are swapped
        img_size = [img_size[1], img_size[0]]

    # assign points to its evaluation array ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
    fhc_points_t.append(test_coordinates[0])
    aF1_points_t.append(test_coordinates[1])
    fnoc_points_t.append(test_coordinates[2])
    tkc_points_t.append(test_coordinates[3])
    sfdma1_points_t.append(test_coordinates[4])
    sfdma2_points_t.append(test_coordinates[5])
    stma1_points_t.append(test_coordinates[6])
    stma2_points_t.append(test_coordinates[7])
    tml_points_t.append(test_coordinates[8])

# The fit learns some quantities from the data, most importantly the "components" and "explained variance":
# Step 1: Fit PCA on training/reference data
num = 2
pca_fhc = PCA(n_components=num).fit(fhc_points_t)
pca_af1 = PCA(n_components=num).fit(aF1_points_t)
pca_fnoc = PCA(n_components=num).fit(fnoc_points_t)
pca_tkc = PCA(n_components=num).fit(tkc_points_t)
pca_sfdma1 = PCA(n_components=num).fit(sfdma1_points_t)
pca_sfdma2 = PCA(n_components=num).fit(sfdma2_points_t)
pca_stma1 = PCA(n_components=num).fit(stma1_points_t)
pca_stma2 = PCA(n_components=num).fit(stma2_points_t)
pca_tml = PCA(n_components=num).fit(tml_points_t)

pca_arr = [pca_fhc, pca_af1, pca_fnoc, pca_tkc, pca_sfdma1, pca_sfdma2, pca_stma1, pca_stma2, pca_tml]
pca_points_arr = [fhc_points_t, aF1_points_t, fnoc_points_t, tkc_points_t, sfdma1_points_t, sfdma2_points_t, stma1_points_t, stma2_points_t, tml_points_t]

# get average point coordinates
pca_average_points = []
for points in pca_points_arr:
    average = yolov8_functions.get_average(points)
    average_point = []
    for p in average:
        average_point.append(p)
    pca_average_points.append(average_point)
#print(pca_average_points)


### Preedictions by YOLO model ###  
directories = yolov8_functions.get_dirs(dataset_path)

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
        missing_labels = []
        i = 0
        conf_box = []
        conf_pose = []

        for result in results:
            for idx, keypoint in enumerate(result.keypoints):
                point = keypoint.xy.tolist()

                x = point[0][0][0]
                y = point[0][0][1]
                landmark = [x,y]

                # get label and point names from result
                label = int(result.boxes.cls[idx])
                label = landmark_names[label]

                """
                if landmark[0] < img_shape[0]*0.15 or landmark[0] > img_shape[0]*0.85:
                    skipped_points.append(landmark)
                    skipped_labels.append(label)
                    continue
                """

                labels.append(label)
                landmarks.append(landmark)
                #conf_box.append(float(result.boxes.conf[idx]))
                #conf_pose.append(float(result.keypoints.conf[idx]))

        #print("Lables:", labels)
        #print("Landmarks:", landmarks)

        # PCA noise filtering and missing coordinate determination 

        # check landmarks for missing labels
        missing = {x for x in landmark_names if labels.count(x) == 0}
        missing = list(missing)
        missing_labels.append(missing)

        # if missing label use second model for specific labels and try to determine ...

        """
        for miss in missing:
            print("Missing:", miss)

            # chose correct PCA method based on duplicate point name
            idx = yolov8_functions.get_indices(miss, landmark_names_pca)

            # PCA choice 
            print("Choosing PCA:", landmark_names_pca[idx[0]])
            pca = pca_arr[idx[0]]

            # Step 2: Transform new coordinates using the fitted PCA model
            predicted_pca = pca.transform([pca_average_points[idx[0]]])
            print("Average " + landmark_names_pca[idx[0]], pca_average_points[idx[0]])
            print("PCA predicted FHC point:", predicted_pca)
        """

        # check landmarks for duplicate labels
        duplicates = {x for x in labels if labels.count(x) > 1}
        duplicates = list(duplicates)

        # get array indexes for duplicates
        duplicate_occurances = []
        removal_arr = []
        for dup in duplicates:
            #print("Duplicate:", dup)
            duplicate_occurances = yolov8_functions.get_indices(dup, labels)

            # check landmarks for duplicate labels
            duplicate_coords = []
            for idx in duplicate_occurances:
                duplicate_coords.append(landmarks[idx])

            # chose correct PCA method based on duplicate point name
            idx = yolov8_functions.get_indices(dup, landmark_names_pca)
            #print("idx:", idx)
            #print("Duplicate point name:", landmark_names_pca[idx[0]])

            # PCA choice 
            #print("Choosing PCA:", landmark_names_pca[idx[0]])
            pca = pca_arr[idx[0]]

            # Step 2: Transform new coordinates using the fitted PCA model
            predicted_pca = pca.transform(duplicate_coords)

            # Step 3: Filter out less accurate points based on Euclidean distance in the reduced space
            center = np.mean(pca.transform(pca_points_arr[idx[0]]), axis=0)
            distances = np.array([euclidean(point, center) for point in predicted_pca])
            #print("Distances:", distances)

            best_idx = 1000000
            best_dist = 1000000
            for idx, dist in enumerate(distances):
                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            to_remove = duplicate_occurances
            del to_remove[best_idx]
            removal_arr.append(to_remove)

            #print("Filtered best idx:", best_idx)
            #print("Best distance:", best_dist)

            # Index the new_coordinates array using the smallest distance index
            filtered_points = duplicate_coords[best_idx]

            # cooridnates for duplicates
            #print("Duplicate coordinates:", duplicate_coords)
            #print("Filtered coordinates:",filtered_points)

            
        # remove other occurances from labels, landmarks, etc
        removal_arr = yolov8_functions.flatten(removal_arr)
        removal_arr.sort(reverse=True)
        #print(removal_arr)
        #print("To remove indexes:", removal_arr)
        for idx in removal_arr:
            removed_labels.append(labels[idx])
            del labels[idx]
            removed_points.append(landmarks[idx])
            del landmarks[idx]
        
        #print("Duplicates:", duplicates)
        #print("Missing:", missing)

        """
        # narejeno v postprocess skripti
        # check for FHC, FNOC & aF1 - dodaj, da ne rabi obstajati aF1 in določi x naknadno glede na y vrednost...
        # x koord je tam kjer je najvišja belost? ker je v sredini kosti
        fhc_p, fnoc_p, aF1_p = False, False, False
        aF1_idx = 0
        for idx, lable in enumerate(labels):
            if lable == 'FHC':
                fhc_p = True
                fhc_idx = idx
            elif lable == 'FNOC':
                fnoc_p = True
                fnoc_idx = idx
            elif lable == "aF1":
                aF1_p = True
                aF1_idx = idx

        # if FHC, aF1 & FNOC exist do aF1 algorithm 
        if (fhc_p and fnoc_p and aF1_p):
            point_aF1 = landmarks[aF1_idx]
            point_fnoc = landmarks[fnoc_idx]
            point_fhc = landmarks[fhc_idx]
            # aF1 algorithm
            af1_y = yolov8_functions.aF1_y_algorithm(point_fnoc, point_fhc)
            aF1 = [point_aF1[0], af1_y]

            # rewrite aF1 point
            landmarks[aF1_idx] = aF1
        """

        for idx, landmark in enumerate(landmarks):
            dictionary.update({
                labels[idx]:landmark,
            })

        dictionary.update({
                "Skipped points":skipped_points,
                "Skipped labels":skipped_labels,
                "Removed points":removed_points,
                "Removed labels":removed_labels,
                "Missing labels":missing_labels
            })

        yolov8_functions.save_prediction_image(landmarks, temp, filename)
        yolov8_functions.create_json_datafile(dictionary, filename)

