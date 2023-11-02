
""" Compare results from original RTG anotations and predicted RTG anotations """
import yolov8_functions
import json
import math

# Dataset path:
json_test_path = "./data/dataset/JSON/"
json_predict_path = "./data/predicted/"
json_save_path = "./data/evaluation"
landmark_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1']

# create dataset archive
yolov8_functions.dataset_archive(json_save_path)

# load Json files

# json paths to predicted folder
json_paths = yolov8_functions.get_dirs(json_predict_path)
json_paths_predicted = []
[json_paths_predicted.append(directory) for directory in json_paths if ".json" in str(directory)]

# remove path before
img_names_predicted =  []
for path in json_paths_predicted:
    image_name = yolov8_functions.filename_creation(path, "")    # change sign="\\" according to linux or windows
    img_names_predicted.append(image_name)

# json paths to test folder
json_paths = yolov8_functions.get_dirs(json_test_path)
json_paths_test = []
for path in json_paths:
    skipLoop = False

    for name in landmark_names:
        if name in path:
            skipLoop = True

    if skipLoop:
        continue

    json_paths_test.append(path)


# get only paths that are to be evaluated from test
json_paths_test_compare = []
for path in json_paths_test:
    skipLoop = True

    for name in img_names_predicted:
        if name in path:
            skipLoop = False

    if skipLoop:
        continue

    json_paths_test_compare.append(path)

# remove path before
img_names_test =  []
img_test_paths = []
for path in json_paths_test_compare:
    image_name = yolov8_functions.filename_creation(path, "")    # change , sign="\\" according to linux or windows
    img_names_test.append(image_name)
    img_test_paths.append(path)

# get only evaluation paths from test paths
to_evaluate_test_paths = []
for idx, img_path in enumerate(img_test_paths):
    if img_names_test[idx] in img_names_predicted:
        to_evaluate_test_paths.append(img_path)
    else:
        continue

for idx, path in enumerate(to_evaluate_test_paths):

    # Test points json
    test_coordinates = 0
    point_names = 0
    img_size = 0
    with open(to_evaluate_test_paths[idx]) as f:
        data = json.load(f)
        test_coordinates = data['Point coordinates']
        point_names = data['Point names']
        img_size =  data['Image_size']  # x,y are swapped

    # Predict points json
    predicted_coordinates = 0
    with open(json_paths_predicted[idx]) as f:
        data = json.load(f)
        predicted_coordinates = data['Point coordinates']

    print("Path:", path)
    #print("Image size:", "[" + str(img_size[1]) + "," + str(img_size[0]) + "]")

    dictionary = {
        "Image name": path,
        "Point names": landmark_names,
        "Image_size": img_size,
        }

    # compare point coordinates
    point_index = 0
    for idx, point in enumerate(test_coordinates):
        
        # compare predicted points to a test point
        for i, x in enumerate(predicted_coordinates):
            coor_y = 1
            coor_x = 0
            percent_y = yolov8_functions.percentage(predicted_coordinates[i][coor_y], test_coordinates[idx][coor_y]) 
            percent_x = yolov8_functions.percentage(predicted_coordinates[i][coor_x], test_coordinates[idx][coor_x]) 

            if percent_y > 90 and percent_y < 110 and percent_x > 90 and percent_x < 110:
                test_point = [test_coordinates[idx][coor_x], test_coordinates[idx][coor_y]]
                predicted_point = [math.ceil(predicted_coordinates[i][coor_x]), math.ceil(predicted_coordinates[i][coor_y])]
                percent_missmatch = [abs(100 - percent_x), abs(100 - percent_y)]
                pixel_error = [abs(test_point[0] - predicted_point[0]), abs(test_point[1] - predicted_point[1])]
                pixel_error_percents = [100*abs((test_point[0] - predicted_point[0])/img_size[0]), 100*abs((test_point[1] - predicted_point[1])/img_size[0])]
                #print("Point name:", point_names[idx])
                #print("Test point X:", test_point[0], "Predicted point X:", predicted_point[0])
                #print("Test point Y:", test_point[1], "Predicted point Y:", predicted_point[1])
                #print("Percentage missmatch X:", "{:.4f}".format(percent_missmatch[0]), "Y:", "{:.4f}".format(percent_missmatch[1]))
                #print("Pixel error X:", "{:.4f}".format(pixel_error[0]), "Y:", "{:.4f}".format(pixel_error[1]))
                #print("Percent pixel error X:", "{:.4f}".format(pixel_error_percents[0]), "Y:", "{:.4f}".format(pixel_error_percents[1])) 
                
                dictionary.update({
                            "Point_" + str(point_index):{
                                "Point_name": point_names[idx],
                                "Test point coordinates": test_point,
                                "Predicted point coordinates": predicted_point,
                                "Percentage missmatch [x,y]": percent_missmatch,
                                "Pixel error [x,y]": pixel_error,
                                "Percent pixel error [x,y]": pixel_error_percents,
                                },
                })
                point_index += 1

    # Save JSON file with data
    filename = json_save_path + "/" + yolov8_functions.filename_creation(path, ".json")
    yolov8_functions.create_json_datafile(dictionary, filename)




