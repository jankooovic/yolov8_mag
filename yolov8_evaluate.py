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
json_postprocess_path = "./data/postprocess/"
json_save_path = "./data/evaluation"
statistics_path = "./data/evaluation/statistics"
slicer_path = "./data/evaluation/slicer_coordinates"
slicerPointTemplate = "./data/evaluation/slicer_coordinates/pointTemplate.mrk.json"
point_names_all = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA', 'sTMA', 'TML']
landmark_names = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
square_size_ratio = 0.1
map_factor = 3.6
coor_y = 1
coor_x = 0

# Arrays
predictedCoord_arr = []
anotatedCoord_arr = []
pixelPercentErr_arr = []
pixelErr_arr = []
missmatchErr_arr = []
eucledian_distances_all = []
eucledian_distances_all_mm = []
skipped = []
evaluated_images = []
mmmErr_arr = []

# Predicted points
aF1_points_p = []
fhc_points_p = []
fnoc_points_p = []
tkc_points_p = []
sfdma1_points_p = []
sfdma2_points_p = []
stma1_points_p = []
stma2_points_p = []
tml_points_p = []

# Test points
aF1_points_t = []
fhc_points_t = []
fnoc_points_t = []
tkc_points_t = []
sfdma1_points_t = []
sfdma2_points_t= []
stma1_points_t = []
stma2_points_t = []
tml_points_t = []

# create dataset archive
yolov8_functions.dataset_archive(json_save_path)

# comment based on what you want
#skipped_path = 'data/postprocess/skipped.json'
skipped_path = 'data/predicted/skipped.json'

# Load json files
json_paths_predicted = [directory for directory in yolov8_functions.get_dirs(json_predict_path) if ".json" in str(directory)]
#json_paths_predicted = [directory for directory in yolov8_functions.get_dirs(json_postprocess_path) if ".json" in str(directory)]
if skipped_path in json_paths_predicted:
    json_paths_predicted.remove(skipped_path)

# get only paths that are to be evaluated from test
json_paths_test = [path for path in yolov8_functions.get_dirs(json_test_path) if not any(name in path for name in point_names_all)]

# get test images
test_images = yolov8_functions.get_dirs(test_images_path)
img_names_test =  [yolov8_functions.filename_creation(path, "") for path in test_images]
for i, img in enumerate(img_names_test):
    img_names_test[i] = img.replace(".jpg","")

# get test json files names
json_names_test =  [yolov8_functions.filename_creation(path, "") for path in json_paths_test]
for i, img in enumerate(json_names_test):
    json_names_test[i] = img.replace(".json","")

# get only evaluation paths from test paths
to_evaluate_json_paths = []
for i, p in enumerate(json_names_test):
    for j, im in enumerate(img_names_test):
        if p == im:
            to_evaluate_json_paths.append(json_paths_test[i])

# remove images with false predictions
to_skip = []
with open(skipped_path) as f:
        data = json.load(f)
        to_skip = (data['False predictions'])

to_skip =  [img_name for img_name in to_skip]
skipping_evaluate = []
skipping_predicted = []
for i, name in enumerate(to_skip):
    to_skip_name = name.replace(".jpg","")
    skipping_evaluate.append("data/dataset/JSON/" + to_skip_name + ".json")
    skipping_predicted.append("data/predicted/" + to_skip_name + ".json")

for skip in skipping_evaluate:
    to_evaluate_json_paths.remove(skip)

for skip in skipping_predicted:
    json_paths_predicted.remove(skip)

# sort paths:
to_evaluate_json_paths = sorted(to_evaluate_json_paths)
json_paths_predicted = sorted(json_paths_predicted)

for idx, path in enumerate(to_evaluate_json_paths):
    skip = False
    print("Path:", path)

    # Test points json
    test_coordinates = []
    point_names = []
    img_size = []
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
                point_names.append('sTMA1')
                test_coordinates.append(stma2)
                point_names.append('sTMA2')
            elif coord == 'sFMDA':
                stma1_x = data[coord][0]
                stma1_y = data[coord][1]
                stma2_x = data[coord][2]
                stma2_y = data[coord][3]
                stma1 = [stma1_x, stma1_y]
                stma2 = [stma2_x, stma2_y]
                test_coordinates.append(stma1)
                point_names.append('sFMDA1')
                test_coordinates.append(stma2)
                point_names.append('sFMDA2')
            else:
                test_coordinates.append(data[coord])
                point_names.append(coord)
        img_size =  data['Image_size']  # x,y are swapped
        img_size = [img_size[1], img_size[0]]

    # Predicted points json
    predicted_coordinates = []
    path = json_paths_predicted[idx]
    print(path)
    with open(path) as f:
        data = json.load(f)
        for name in landmark_names:
            predicted_coordinates.append(data[name])

    dictionary = {
        "Image name": path,
        "Point names": point_names_all,
        "Image_size": img_size,
        }
    
    #print("Test coordinates     :", test_coordinates)
    #print("Predicted coordinates:", predicted_coordinates)

    # assign points to its evaluation array ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
    aF1_points_t.append(test_coordinates[1])
    fhc_points_t.append(test_coordinates[0])
    fnoc_points_t.append(test_coordinates[2])
    tkc_points_t.append(test_coordinates[3])
    sfdma1_points_t.append(test_coordinates[4])
    sfdma2_points_t.append(test_coordinates[5])
    stma1_points_t.append(test_coordinates[6])
    stma2_points_t.append(test_coordinates[7])
    tml_points_t.append(test_coordinates[8])

    aF1_points_p.append(predicted_coordinates[1])
    fhc_points_p.append(predicted_coordinates[0])
    fnoc_points_p.append(predicted_coordinates[2])
    tkc_points_p.append(predicted_coordinates[3])
    sfdma1_points_p.append(predicted_coordinates[4])
    sfdma2_points_p.append(predicted_coordinates[5])
    stma1_points_p.append(predicted_coordinates[6])
    stma2_points_p.append(predicted_coordinates[7])
    tml_points_p.append(predicted_coordinates[8])

    # check for missing coordinates
    for idx, point in enumerate(test_coordinates):

        # check if points were predicted correctly
        
        percent_y = yolov8_functions.percentage(predicted_coordinates[idx][coor_y], test_coordinates[idx][coor_y]) 
        percent_x = yolov8_functions.percentage(predicted_coordinates[idx][coor_x], test_coordinates[idx][coor_x]) 
        
        if (90 > percent_y):
            text = "Predicted coordinate missmatch:"
            skip = True
        elif(110 < percent_y):
            text = "Predicted coordinate missmatch:"
            skip = True
        elif(110 < percent_x):
            text = "Predicted coordinate missmatch:"
            skip = True
        elif(90 > percent_x):
            text = "Predicted coordinate missmatch:"
            skip = True

    if (skip):
        #print(text, path)
        skipped.append(path)
        #continue
    
    # compare point coordinates
    for idx, point in enumerate(test_coordinates):

        percent_y = yolov8_functions.percentage(predicted_coordinates[idx][coor_y], test_coordinates[idx][coor_y]) 
        percent_x = yolov8_functions.percentage(predicted_coordinates[idx][coor_x], test_coordinates[idx][coor_x]) 

        test_point = [test_coordinates[idx][coor_x], test_coordinates[idx][coor_y]]
        predicted_point = [predicted_coordinates[idx][coor_x], predicted_coordinates[idx][coor_y]]
        percent_missmatch = [abs(100 - percent_x), abs(100 - percent_y)]
        pixel_error = [abs(test_point[0] - predicted_point[0]), abs(test_point[1] - predicted_point[1])]
        pixel_error_percents = [100*abs((test_point[0] - predicted_point[0])/img_size[0]), 100*abs((test_point[1] - predicted_point[1])/img_size[0])] 
        eucledian_distance = yolov8_functions.euclidean_distance(predicted_coordinates[idx], test_coordinates[idx])

        missmatchErr_arr.append(percent_missmatch)
        pixelErr_arr.append(pixel_error)
        pixelPercentErr_arr.append(pixel_error_percents)

        predictedCoord_arr.append(predicted_point)
        anotatedCoord_arr.append(test_point)
        eucledian_distances_all.append(eucledian_distance)
        
        dictionary.update({
                    landmark_names[idx]:{
                        "Test point coordinates [x,y]": test_point,
                        "Predicted point coordinates [x,y]": predicted_point,
                        "Percentage missmatch [x,y]": percent_missmatch,
                        "Pixel error [x,y]": pixel_error,
                        "mm error [x,y]": [pixel_error[0] / 3.6, pixel_error[1] / 3.6],
                        "Percent pixel error [x,y]": pixel_error_percents,
                        "Eucledian distance pixel": eucledian_distance,
                        "Eucledian distance mm": eucledian_distance / 3.6,
                        },
        })

        # Open and save Slicer point template
        with open(slicerPointTemplate) as f:
            data = json.load(f)
        data['markups'][0]['controlPoints'][0]['label'] = landmark_names[idx]
        data['markups'][0]['controlPoints'][0]['position'] = [predicted_point[0]/map_factor,0.1,-predicted_point[1]/map_factor]
        point_name = yolov8_functions.filename_creation(path, ".json")
        point_filename = slicer_path + "/" + point_name + "_" + landmark_names[idx] + ".mrk"
        yolov8_functions.create_json_datafile(data, point_filename)
        f.close()

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
                    im = np.array(image)[round(p[1]-half_side):round(p[1]+half_side),round(p[0]-half_side):round(p[0]+half_side)]
        
                    fig, ax = plt.subplots()
                    ax.plot(half_side, half_side, marker='x', color="black")  # test point
                    ax.plot(half_side + (p[0] - pp[0]),half_side + (p[1] - pp[1]), marker='+', color="red")  # predicted point

                    plt.imshow(im, cmap="gray")
                    plt.savefig(filename + "_" + landmark_names[i] + '.png')
                    #plt.show()
                    plt.close()

if (len(predictedCoord_arr) != 0):
    # Error statistics - explanation in -/documents/graphs_explanation.txt
    for i in pixelErr_arr:
        x = i[coor_x] / map_factor
        y = i[coor_y] / map_factor
        p = [x,y]
        mmmErr_arr.append(p)

    for i in eucledian_distances_all:
        x = i / map_factor
        eucledian_distances_all_mm.append(x)

    dictionary = {
        "Average pixel error [x,y]": yolov8_functions.get_average(pixelErr_arr),
        "Average mm error [x,y]": yolov8_functions.get_average(mmmErr_arr),
        "Average euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_distances_all), yolov8_functions.get_average_one(eucledian_distances_all_mm)],
        "Average FHC error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(fhc_points_p, fhc_points_t),
        "Average aF1 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(aF1_points_p, aF1_points_t),
        "Average FNOC error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(fnoc_points_p, fnoc_points_t),
        "Average TKC error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(tkc_points_p, tkc_points_t),
        "Average sFDMA1 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(sfdma1_points_p, sfdma1_points_t),
        "Average sFDMA2 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(sfdma2_points_p, sfdma2_points_t),
        "Average sTMA1 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(stma1_points_p, stma1_points_t),
        "Average sTMA2 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(stma2_points_p, stma2_points_t),
        "Average TML error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(tml_points_p, tml_points_t),
        "Images with >10 percent error": skipped,
        "False predictions": to_skip,
        "Average pixel error percentage [x,y]": yolov8_functions.get_average(pixelPercentErr_arr),
        "Average missmatch error [x,y]": yolov8_functions.get_average(missmatchErr_arr),
    }

    # Save JSON file with data
    filename = statistics_path + "/" + "errors"
    yolov8_functions.create_json_datafile(dictionary, filename)

    measured_data_x, measured_data_y = yolov8_functions.extract_points(predictedCoord_arr)
    true_data_x, true_data_y = yolov8_functions.extract_points(anotatedCoord_arr)

    # x coordinate
    yolov8_functions.scatter_plot(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.residual_plot(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.histogram_of_errors(abs(true_data_x - measured_data_x), "X", statistics_path)
    yolov8_functions.qq_plot(abs(true_data_x - measured_data_x), "X", statistics_path)
    yolov8_functions.bland_altman_plot(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.box_plot(abs(true_data_x - measured_data_x), "X", statistics_path)
    yolov8_functions.heatmap(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.violin_plot_of_differences(measured_data_x, true_data_x, "X", statistics_path)

    diff_range, diff_iqr = yolov8_functions.range_and_iqr_of_differences(measured_data_x, true_data_x)
    diff_std_dev = yolov8_functions.standard_deviation_of_differences(measured_data_x, true_data_x)
    diff_cv = yolov8_functions.coefficient_of_variation_of_differences(measured_data_x, true_data_x)

    dictionary = {
        "Difference range" : float(diff_range),
        "Difference IQR (Interquartile Range)" : float(diff_iqr),
        "Difference Standard Deviation" : float(diff_std_dev),
        "Difference Coefficient of Variation" : float(diff_cv),
    }

    # Save JSON file with data
    filename = statistics_path + "/" + "variability_X"
    yolov8_functions.create_json_datafile(dictionary, filename)

    # y coordinate
    yolov8_functions.scatter_plot(measured_data_y, true_data_y, "Y", statistics_path)
    yolov8_functions.residual_plot(measured_data_y, true_data_y, "Y", statistics_path)
    yolov8_functions.histogram_of_errors(abs(true_data_y - measured_data_y), "Y", statistics_path)
    yolov8_functions.qq_plot(abs(true_data_y - measured_data_y), "Y", statistics_path)
    yolov8_functions.bland_altman_plot(measured_data_y, true_data_y, "Y", statistics_path)
    yolov8_functions.box_plot(abs(true_data_y - measured_data_y), "Y", statistics_path)
    yolov8_functions.heatmap(measured_data_y, true_data_y, "Y", statistics_path)
    yolov8_functions.violin_plot_of_differences(measured_data_y, true_data_y, "Y", statistics_path)

    diff_range, diff_iqr = yolov8_functions.range_and_iqr_of_differences(measured_data_y, true_data_y)
    diff_std_dev = yolov8_functions.standard_deviation_of_differences(measured_data_y, true_data_y)
    diff_cv = yolov8_functions.coefficient_of_variation_of_differences(measured_data_y, true_data_y)

    dictionary = {
        "Difference range" : float(diff_range),
        "Difference IQR (Interquartile Range)" : float(diff_iqr),
        "Difference Standard Deviation" : float(diff_std_dev),
        "Difference Coefficient of Variation" : float(diff_cv),
    }

    # Save JSON file with data
    filename = statistics_path + "/" + "variability_Y"
    yolov8_functions.create_json_datafile(dictionary, filename)

    # Points plots
    test_arrs = [fhc_points_t, aF1_points_t, fnoc_points_t, tkc_points_t, sfdma1_points_t, sfdma2_points_t, stma1_points_t, stma2_points_t, tml_points_t]
    predicted_arrs = [fhc_points_p, aF1_points_p, fnoc_points_p, tkc_points_p, sfdma1_points_p, sfdma2_points_p, stma1_points_p, stma2_points_p, tml_points_p]
    for idx, name in enumerate(landmark_names):

        test_data_x, test_data_y = yolov8_functions.extract_points(test_arrs[idx])
        predicted_data_x, predicted_data_y = yolov8_functions.extract_points(predicted_arrs[idx])
        yolov8_functions.box_plot(abs(test_data_x - predicted_data_x), name + " X", statistics_path)
        yolov8_functions.box_plot(abs(test_data_y - predicted_data_y), name + " Y", statistics_path)