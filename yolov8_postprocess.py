""" Postprocess predictions """
from ultralytics import YOLO
import cv2
import numpy as np
import yolov8_functions
import json
import matplotlib.pyplot as plt

# Dataset path:
predicted_path = "./data/predicted/"
postprocess_path = "./data/postprocess/"
save_path = "./data/postprocess"
images_path = "./data/dataset/ALL/images/test/"
landmark_names = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
skipped_path = "data/predicted/skipped.json"
image_name = None

# create dataset archive
yolov8_functions.dataset_archive(save_path)

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
    img_shape = None
    with open(json_paths_predicted[idx]) as f:
        data = json.load(f)
        img_shape = data['Image_size']
        image_name = data['Image name']
        for name in landmark_names:
            predicted_coordinates.append(data[name])

    #print("Image path:", img_path, "Point:", json_paths_predicted[idx])
    #print("Predicted coordinates:", predicted_coordinates)

    # load image
    image = cv2.imread(img_path)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 20, 80)

    # Define a kernel for dilation and erosion
    kernel = np.ones((3, 3), np.uint8)

    # Dilate the edges to connect them and make them thicker
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Erode the edges to make them thinner and smoother
    smoothed_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the points to find on edges ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
    point_fhc = predicted_coordinates[0]
    point_aF1 = predicted_coordinates[1]
    point_fnoc = predicted_coordinates[2]
    point_tkc = predicted_coordinates[3]
    point_sfdma1 = predicted_coordinates[4]
    point_sfdma2 = predicted_coordinates[5]
    point_stma1 = predicted_coordinates[6]
    point_stma2 = predicted_coordinates[7]
    point_tml = predicted_coordinates[8]

    closest_contour_fnoc, point_on_contour_fnoc = yolov8_functions.find_point_on_contour(contours, point_fnoc)

    # update fnoc coordinate 
    predicted_coordinates[2] = point_on_contour_fnoc

    """
    # Display the original image with closest contour
    image_with_closest_contour = image.copy()
    #cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(image_with_closest_contour, [closest_contour_fnoc], -1, (0, 255, 0), 2)
    
    # plot predicted points
    fig, ax = plt.subplots()   
    for point in predicted_coordinates: 
        ax.plot(*point, marker='o', color="white")
    plt.imshow(image_with_closest_contour, cmap="gray")
    plt.show()
    """

    # aF1 algorithm
    af1_y = yolov8_functions.aF1_y_algorithm(point_on_contour_fnoc, point_fhc)
    aF1 = [point_aF1[0], af1_y]

    # update fnoc coordinate 
    predicted_coordinates[1] = aF1

    """
    # show results
    print("Point FNOC on contour: ", point_on_contour_fnoc)
    print("Point FNOC: ", point_fnoc)
    print("Point aF1 algorithm:", aF1)
    print("Point aF1:", point_aF1)
    """

    # Save JSON file with data
    dictionary = {
        "Image name": image_name,
        "Image_size": img_shape,
    }

    for idx, landmark in enumerate(predicted_coordinates):
        landmark[0] = float(landmark[0])
        landmark[1] = float(landmark[1])
        dictionary.update({
            landmark_names[idx]:landmark,
        })

    filename = postprocess_path + image_name
    yolov8_functions.create_json_datafile(dictionary, filename)

# Find midlle of edged for TML
# Allign TKC, FHC better? 
