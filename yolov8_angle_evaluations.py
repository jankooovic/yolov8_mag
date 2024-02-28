""" Compare results from original RTG anotations and predicted RTG anotations """
import yolov8_functions
import json
import math
import matplotlib.pyplot as plt
import numpy as np


def calculate_slope(point1, point2):
    # Calculate the slope (m) between two points
    x1, y1 = point1
    x2, y2 = point2
    return (y2 - y1) / (x2 - x1)

def calculate_angle(point1, point2, point3, point4):
    # Calculate slopes for both lines
    slope1 = calculate_slope(point1, point2)
    slope2 = calculate_slope(point3, point4)

    # Calculate the angle between the slopes
    angle_radians = math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2)))

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees




# Dataset path:
test_images_path =  "./data/dataset/ALL/images/test/"
json_test_path = "./data/dataset/JSON/"
json_predict_path = "./data/predicted/"
json_postprocess_path = "./data/postprocess/"
json_save_path = "./data/evaluation"
statistics_path = "./data/evaluation/statistics"
slicer_path = "./data/evaluation/slicer_coordinates"
angles_path = "./data/evaluation/angles"
json_errors = "./data/evaluation/statistics/errors.json"
slicerPointTemplate = "./data/evaluation/slicer_coordinates/pointTemplate.mrk.json"
point_names_all = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA', 'sTMA', 'TML']
landmark_names = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
square_size_ratio = 0.1
map_factor = 3.6
coor_y = 1
coor_x = 0

all_HKA_test = []
all_HKA_predicted = []
all_FSTS_test = []
all_FSTS_predicted = []


# search for files with points
json_file_paths = [directory for directory in yolov8_functions.get_dirs(json_save_path) if ".json" in str(directory)]
print(json_file_paths)

# get points 
for idx, path in enumerate(json_file_paths):
    print("Path:", path)

    # Test points json
    test_coordinates = []
    predicted_coordinates = []
    point_names = []
    img_size = []
    img_name = ""
    with open(path) as f:
        data = json.load(f)
        for coord in landmark_names:
            test_coordinates.append(data[coord]["Test point coordinates [x,y]"])
            predicted_coordinates.append(data[coord]["Predicted point coordinates [x,y]"])
            point_names.append(coord)
        img_size = data['Image_size']  # x,y are swapped
        img_size = [img_size[1], img_size[0]]
        img_name = data['Image name']
    print("")
    print("Name: ", img_name)
    print("Size: ", img_size)
    print("Points: ", test_coordinates)
    print("Points: ", predicted_coordinates)

    # calculate angles - ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
    angle_HKA_test = calculate_angle(test_coordinates[0], test_coordinates[2], test_coordinates[3], test_coordinates[8])
    angle_HKA_predicted = calculate_angle(predicted_coordinates[0], predicted_coordinates[2], predicted_coordinates[3], predicted_coordinates[8])
    print(f"The HKA angle is: {angle_HKA_test:.2f} degrees.")
    print(f"The HKA angle is: {angle_HKA_predicted:.2f} degrees.")

    angle_FSTS_test = calculate_angle(test_coordinates[1], test_coordinates[2], test_coordinates[3], test_coordinates[8])
    angle_FSTS_predicted = calculate_angle(predicted_coordinates[1], predicted_coordinates[2], predicted_coordinates[3], predicted_coordinates[8])
    print(f"The FS-TS angle is: {angle_FSTS_test:.2f} degrees.")
    print(f"The FS-TS angle is: {angle_FSTS_predicted:.2f} degrees.")

    all_HKA_test.append(angle_HKA_test)
    all_HKA_predicted.append(angle_HKA_predicted)
    all_FSTS_test.append(angle_FSTS_test)
    all_FSTS_predicted.append(angle_FSTS_predicted)
    
    # write data to json file
    dictionary = {
        "Image name": path,
        "Point names": point_names_all,
        "Image_size": img_size,
        "Test points": test_coordinates,
        "Predicted points": predicted_coordinates,
        "HKA angle test": angle_HKA_test,
        "HKA angle predicted": angle_HKA_predicted,
        "HKA diff": abs(np.array(angle_HKA_test) - np.array(angle_HKA_predicted)),
        "FS-TS angle test": angle_FSTS_test,
        "FS-TS angle predicted": angle_FSTS_predicted,
        "FS-TS diff": abs(np.array(angle_FSTS_test) - np.array(angle_FSTS_predicted)),
        }
    
    # Save JSON file with data
    name = yolov8_functions.filename_creation(path, ".json")
    filename = angles_path + "/" + name
    yolov8_functions.create_json_datafile(dictionary, filename)

# calculate the averages and min max values
average_HKA_test = yolov8_functions.get_average_one(all_HKA_test)
average_HKA_predicted = yolov8_functions.get_average_one(all_HKA_predicted)
average_FSTS_test = yolov8_functions.get_average_one(all_FSTS_test)
average_FSTS_predicted = yolov8_functions.get_average_one(all_FSTS_predicted)

diff_HKA = abs(np.array(all_HKA_test) - np.array(all_HKA_predicted))
diff_FSTS = abs(np.array(all_FSTS_test) - np.array(all_FSTS_predicted))

# create a list of all images where predictions were satisfactory
succ_localization = []
print("")
for idx, path in enumerate(json_file_paths):
    if diff_HKA[idx] < 3:
        name = name = yolov8_functions.filename_creation(path, ".json")
        print("Successfull localization on image: ", name)
        succ_localization.append(name)

dictionary = {
    "HKA angle test average": average_HKA_test,
    "HKA angle predicted average": average_HKA_predicted,
    "HKA angle min/max diff": [min(diff_HKA),max(diff_HKA)],
    "FS-TS angle test average": average_FSTS_test,
    "FS-TS angle predicted average": average_FSTS_predicted,
    "FS-TS angle min/max diff": [min(diff_FSTS),max(diff_FSTS)],
    "Successfull localization": succ_localization 
    }

# Save JSON file with data
filename = angles_path + "/" + "errors"
yolov8_functions.create_json_datafile(dictionary, filename)