""" Compare results from original RTG anotations and predicted RTG anotations """
import yolov8_functions
import json
import math
import matplotlib.pyplot as plt
import numpy as np

# Dataset path:
test_images_path =  "./data/dataset/ALL/images/test/"
json_test_path = "./data/dataset/JSON/"
json_predict_path = "./data/predicted/"
json_save_path = "./data/evaluation"
landmark_names = ['FHC', 'aF1', 'TKC', 'FNOC', 'TML']
square_size_ratio = 0.1
missmatchErr_arr = []
pixelErr_arr = []
pixelPercentErr_arr = []

# create dataset archive
yolov8_functions.dataset_archive(json_save_path)

# Load json files
json_paths_predicted = [directory for directory in yolov8_functions.get_dirs(json_predict_path) if ".json" in str(directory)]
img_names_predicted =  [yolov8_functions.filename_creation(path, "") for path in json_paths_predicted] # change sign="\\" according to linux or windows

# get only paths that are to be evaluated from test
json_paths_test = [path for path in yolov8_functions.get_dirs(json_test_path) if not any(name in path for name in landmark_names)]
json_paths_test_compare = [path for path in json_paths_test if any(name in path for name in img_names_predicted)]
img_names_test =  [yolov8_functions.filename_creation(path, "") for path in json_paths_test_compare] # change sign="\\" according to linux or windows
img_test_paths = [path for path in json_paths_test_compare]

# get only evaluation paths from test paths
to_evaluate_test_paths = [img_path for idx, img_path in enumerate(img_test_paths) if img_names_test[idx] in img_names_predicted]

# get test images
test_images = yolov8_functions.get_dirs(test_images_path)

for idx, path in enumerate(to_evaluate_test_paths):

    # Test points json
    test_coordinates = 0
    point_names = 0
    img_size = 0
    with open(path) as f:
        data = json.load(f)
        test_coordinates = data['Point coordinates']
        point_names = data['Point names']
        img_size =  data['Image_size']  # x,y are swapped

    # Predicted points json
    predicted_coordinates = 0
    with open(json_paths_predicted[idx]) as f:
        data = json.load(f)
        predicted_coordinates = data['Point coordinates']

    print("Path:", path)
    dictionary = {
        "Image name": path,
        "Point names": landmark_names,
        "Image_size": img_size,
        }
    # sort points based on y coordinates [FHC, aF1, TKC, FNOC, TML]
    test_coordinates = sorted(test_coordinates, key=lambda point: point[1])
    predicted_coordinates = sorted(predicted_coordinates, key=lambda point: point[1])

    # compare point coordinates
    for idx, point in enumerate(test_coordinates):
        coor_y = 1
        coor_x = 0
        percent_y = yolov8_functions.percentage(predicted_coordinates[idx][coor_y], test_coordinates[idx][coor_y]) 
        percent_x = yolov8_functions.percentage(predicted_coordinates[idx][coor_x], test_coordinates[idx][coor_x]) 

        test_point = [test_coordinates[idx][coor_x], test_coordinates[idx][coor_y]]
        predicted_point = [math.ceil(predicted_coordinates[idx][coor_x]), math.ceil(predicted_coordinates[idx][coor_y])]
        percent_missmatch = [abs(100 - percent_x), abs(100 - percent_y)]
        pixel_error = [abs(test_point[0] - predicted_point[0]), abs(test_point[1] - predicted_point[1])]
        pixel_error_percents = [100*abs((test_point[0] - predicted_point[0])/img_size[0]), 100*abs((test_point[1] - predicted_point[1])/img_size[0])] 

        missmatchErr_arr.append(percent_missmatch)
        pixelErr_arr.append(pixel_error)
        pixelPercentErr_arr.append(pixel_error_percents)
        
        dictionary.update({
                    landmark_names[idx]:{
                        "Test point coordinates": test_point,
                        "Predicted point coordinates": predicted_point,
                        "Percentage missmatch [x,y]": percent_missmatch,
                        "Pixel error [x,y]": pixel_error,
                        "Percent pixel error [x,y]": pixel_error_percents,
                        },
        })

    # Save JSON file with data
    name = yolov8_functions.filename_creation(path, ".json")
    filename = json_save_path + "/" + name
    yolov8_functions.create_json_datafile(dictionary, filename)

    # open image based on point name
    for img in test_images:
        if "./" + img == test_images_path + name + ".jpg":
            
            image = yolov8_functions.open_image(img)
            yolov8_functions.save_evaluation_image(image, filename, test_coordinates, predicted_coordinates)
            image_shape = np.array(image).shape # x and y are switched
            square_side = image_shape[0]*square_size_ratio
            half_side = math.ceil(square_side/2)

            for i, p in enumerate(test_coordinates):
                    coor_y = 1
                    coor_x = 0
                    pp = predicted_coordinates[i]
                    percent_y = yolov8_functions.percentage(pp[coor_y], p[coor_y])
                    percent_x = yolov8_functions.percentage(pp[coor_x], p[coor_x])

                    p = np.array(p)
                    im = np.array(image)[p[1]-half_side:p[1]+half_side,p[0]-half_side:p[0]+half_side]
        
                    fig, ax = plt.subplots()
                    ax.plot(half_side, half_side, marker='x', color="black")  # test point
                    ax.plot(half_side + (p[0] - pp[0]),half_side + (p[1] - pp[1]), marker='+', color="red")  # predicted point

                    plt.imshow(im, cmap="gray")
                    plt.savefig(filename + "_" + landmark_names[i] + '.png')
                    #plt.show()
                    plt.close()

dictionary = {
    "Average missmatch error": yolov8_functions.get_average(missmatchErr_arr),
    "Average pixel error": yolov8_functions.get_average(pixelErr_arr),
    "Average pixel error percentage": yolov8_functions.get_average(pixelPercentErr_arr),
}

# Save JSON file with data
filename = json_save_path + "/" + "errors.json"
yolov8_functions.create_json_datafile(dictionary, filename)

