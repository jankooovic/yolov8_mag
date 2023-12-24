""" Compare RTG anotations between two specialists """
import yolov8_functions
import math
import pathlib

# Dataset path:
dataset_path_igor =  "./data/RTG_dataset_Igor/"
dataset_path_andrej =  "./data/RTG_dataset_Andrej/"
save_path = "./data/evaluate_specialists"
statistics_path = "./data/evaluate_specialists/statistics"
point_names_all = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA', 'sTMA', 'TML']
landmark_names = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sTMA1', 'sFMDA2', 'sTMA2','TML']
square_size_ratio = 0.1
map_factor = 3.6
predictedCoord_arr, anotatedCoord_arr, pixelPercentErr_arr, pixelErr_arr, missmatchErr_arr, skipped, evaluated_images, mmmErr_arr = [], [], [], [], [], [], [], []
coor_y = 1
coor_x = 0
filter_val = 10000

# create dataset archive
yolov8_functions.dataset_archive(save_path)
                                 
# Load json files
paths_igor = yolov8_functions.get_dirs(dataset_path_igor)
paths_andrej = yolov8_functions.get_dirs(dataset_path_andrej)

paths_to_compare = []
for i_path in paths_igor:
    for a_path in paths_andrej:
        i_p = i_path.replace("data/RTG_dataset_Igor/", "")
        a_p = a_path.replace("data/RTG_dataset_Andrej/", "")
        if i_p == a_p: 
            paths_to_compare.append(i_path)
            paths_to_compare.append(a_path)


# For each folder open json files and compare coordinates
for idx, path in enumerate(paths_to_compare):
    if idx%2 == 0:
        path_i = paths_to_compare[idx]
        path_a = paths_to_compare[idx+1]

        json_files_i = [directory for directory in yolov8_functions.get_dirs(path_i) if ".json" in str(directory)]
        json_files_a = [directory for directory in yolov8_functions.get_dirs(path_a) if ".json" in str(directory)]

        # open json files
        points_i = yolov8_functions.create_point_array(json_files_i, map_factor)
        points_a = yolov8_functions.create_point_array(json_files_a, map_factor)

        
        # sort points based on Y&X coordinates [FHC, aF1, TKC, FNOC, sFMDA, sTMA, TML] 
        points_i = sorted(points_i, key=lambda point: point[1])
        points_a = sorted(points_a, key=lambda point: point[1])

        # sort knee Points
        igor_sPoints = points_i[2:8]
        andrej_sPoints = points_a[2:8]
        # sort the 6 points based on X values
        igor_sPoints = sorted(igor_sPoints, key=lambda point: point[0])
        andrej_sPoints = sorted(andrej_sPoints, key=lambda point: point[0])
        
        igor_sPoints = yolov8_functions.sort_sPoints(igor_sPoints)
        andrej_sPoints = yolov8_functions.sort_sPoints(andrej_sPoints)

        points_i[2:8] = igor_sPoints
        points_a[2:8] = andrej_sPoints

        # remove af1 
        del points_i[1]
        del points_a[1]

        print("Slika:", path_i)
        #print("Igor točke:  ", points_i)
        #print("Andrej točke:", points_a)

        # get image size
        nrrd_image_path = 0
        for idx, item in enumerate(pathlib.Path(path + "/").iterdir()):
            item = str(item)
            if ".nrrd" in item:
                data_arr, img_size, orig_img_ratio = yolov8_functions.preprocess_image(item, filter_val)

        dictionary = {
            "Image name": path,
            "Point names": point_names_all,
            "Image size": img_size
            }
        
        # compare point coordinates
        for idx, point in enumerate(points_i):

            percent_y = yolov8_functions.percentage(points_a[idx][coor_y], points_i[idx][coor_y]) 
            percent_x = yolov8_functions.percentage(points_a[idx][coor_x], points_i[idx][coor_x]) 

            test_point = [points_i[idx][coor_x], points_i[idx][coor_y]]
            predicted_point = [math.ceil(points_a[idx][coor_x]), math.ceil(points_a[idx][coor_y])]
            percent_missmatch = [abs(100 - percent_x), abs(100 - percent_y)]
            pixel_error = [abs(test_point[0] - predicted_point[0]), abs(test_point[1] - predicted_point[1])]
            pixel_error_percents = [100*abs((test_point[0] - predicted_point[0])/img_size[0]), 100*abs((test_point[1] - predicted_point[1])/img_size[0])] 

            missmatchErr_arr.append(percent_missmatch)
            pixelErr_arr.append(pixel_error)
            pixelPercentErr_arr.append(pixel_error_percents)

            predictedCoord_arr.append(predicted_point)
            anotatedCoord_arr.append(test_point)
            
            dictionary.update({
                        landmark_names[idx]:{
                            "Test point coordinates [x,y]": test_point,
                            "Predicted point coordinates [x,y]": predicted_point,
                            "Percentage missmatch [x,y]": percent_missmatch,
                            "Pixel error [x,y]": pixel_error,
                            "Percent pixel error [x,y]": pixel_error_percents,
                            },
            })

        # Save JSON file with data
        name = yolov8_functions.filename_creation(path, ".json")
        filename = save_path + "/" + name
        yolov8_functions.create_json_datafile(dictionary, filename)

if (len(predictedCoord_arr) != 0):
    # Error statistics - explanation in -/documents/graphs_explanation.txt
    for i in pixelErr_arr:
        x = math.ceil(i[coor_x]/ map_factor)
        y = math.ceil(i[coor_y]/ map_factor)
        p = [x,y]
        mmmErr_arr.append(p)

    dictionary = {
        "Average missmatch error [x,y]": yolov8_functions.get_average(missmatchErr_arr),
        "Average pixel error [x,y]": yolov8_functions.get_average(pixelErr_arr),
        "Average mm error [x,y]": yolov8_functions.get_average(mmmErr_arr),
        "Average pixel error percentage [x,y]": yolov8_functions.get_average(pixelPercentErr_arr),
        "Skipped images:": skipped
    }

    # Save JSON file with data
    filename = statistics_path + "/" + "errors"
    yolov8_functions.create_json_datafile(dictionary, filename)

    measured_data_x, measured_data_y = yolov8_functions.extract_points(predictedCoord_arr)
    true_data_x, true_data_y = yolov8_functions.extract_points(anotatedCoord_arr)

    # x coordinate
    yolov8_functions.scatter_plot(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.residual_plot(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.histogram_of_errors(true_data_x - measured_data_x, "X", statistics_path)
    yolov8_functions.qq_plot(true_data_x - measured_data_x, "X", statistics_path)
    yolov8_functions.bland_altman_plot(measured_data_x, true_data_x, "X", statistics_path)
    yolov8_functions.box_plot(true_data_x - measured_data_x, "X", statistics_path)
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
    yolov8_functions.histogram_of_errors(true_data_y - measured_data_y, "Y", statistics_path)
    yolov8_functions.qq_plot(true_data_y - measured_data_y, "Y", statistics_path)
    yolov8_functions.bland_altman_plot(measured_data_y, true_data_y, "Y", statistics_path)
    yolov8_functions.box_plot(true_data_y - measured_data_y, "Y", statistics_path)
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
