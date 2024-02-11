""" Compare RTG anotations between two specialists """
import yolov8_functions
import math
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Dataset path:
dataset_path_igor =  "./data/RTG_dataset_Igor/"
dataset_path_andrej =  "./data/RTG_dataset_Andrej/"
save_path = "./data/evaluate_specialists"
statistics_path = "./data/evaluate_specialists/statistics"
point_names_all = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA', 'sTMA', 'TML']
landmark_names = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sTMA1', 'sFMDA2', 'sTMA2','TML']
landmark_names_no_aF1 = ['FHC', 'FNOC', 'TKC', 'sFMDA1', 'sTMA1', 'sFMDA2', 'sTMA2','TML']
square_size_ratio = 0.1
map_factor = 3.6
predictedCoord_arr, anotatedCoord_arr, pixelPercentErr_arr, pixelErr_arr, missmatchErr_arr, skipped, evaluated_images = [], [], [], [], [], [], []
coor_y = 1
coor_x = 0
filter_val = 10000

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

# Igor points
aF1_points_p = []
fhc_points_p = []
fnoc_points_p = []
tkc_points_p = []
sfdma1_points_p = []
sfdma2_points_p = []
stma1_points_p = []
stma2_points_p = []
tml_points_p = []

# Andrej points
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

        # sort json files based on points
        # FHC, FNOC, TKC, TML, aF1, sFMDA, sTMA
        json_files_i = sorted(json_files_i)
        json_files_a = sorted(json_files_a)

        #print("Igor json:", json_files_i)
        #print("Andrej json:", json_files_a)

        # open json files
        points_i = yolov8_functions.create_point_array(json_files_i, map_factor)
        points_a = yolov8_functions.create_point_array(json_files_a, map_factor)

        

        # remove af1 
        del points_i[4]
        del points_a[4]

        print(points_i)

        # FHC, FNOC, TKC, TML, sFMDA, sTMA# FHC, FNOC, TKC, TML, sFMDA, sTMA
        print("Slika:", path_i)
        #print("Igor točke:  ", points_i, len(points_i))
        #print("Andrej točke:", points_a, len(points_a))

        # get image size
        img_size = None
        image = None
        for idx, item in enumerate(pathlib.Path(path + "/").iterdir()):
            item = str(item)
            if ".nrrd" in item:
                data_arr, img_size, orig_img_ratio = yolov8_functions.preprocess_image(item, filter_val)
                image = data_arr
        dictionary = {
            "Image name": path,
            "Point names": point_names_all,
            "Image size": img_size
            }

        # assign points to its evaluation array # FHC, FNOC, TKC, TML, sFMDA, sTMA
        fhc_points_t.append(points_a[0])
        fnoc_points_t.append(points_a[1])
        tkc_points_t.append(points_a[2])
        sfdma1_points_t.append(points_a[4])
        sfdma2_points_t.append(points_a[5])
        stma1_points_t.append(points_a[6])
        stma2_points_t.append(points_a[7])
        tml_points_t.append(points_a[3])

        fhc_points_p.append(points_i[0])
        fnoc_points_p.append(points_i[1])
        tkc_points_p.append(points_i[2])
        sfdma1_points_p.append(points_i[4])
        sfdma2_points_p.append(points_i[5])
        stma1_points_p.append(points_i[6])
        stma2_points_p.append(points_i[7])
        tml_points_p.append(points_i[3])
        
        # compare point coordinates
        for idx, point in enumerate(points_i):

            percent_y = yolov8_functions.percentage(points_a[idx][coor_y], points_i[idx][coor_y]) 
            percent_x = yolov8_functions.percentage(points_a[idx][coor_x], points_i[idx][coor_x]) 
            # print("Percent Y:", percent_y, "Percent X:", percent_x)

            test_point = [points_i[idx][coor_x], points_i[idx][coor_y]]
            predicted_point = [points_a[idx][coor_x], points_a[idx][coor_y]]
            percent_missmatch = [abs(100 - percent_x), abs(100 - percent_y)]
            pixel_error = [abs(test_point[0] - predicted_point[0]), abs(test_point[1] - predicted_point[1])]
            pixel_error_percents = [100*abs((test_point[0] - predicted_point[0])/img_size[0]), 100*abs((test_point[1] - predicted_point[1])/img_size[0])] 
            eucledian_distance = yolov8_functions.euclidean_distance(points_i[idx], points_a[idx])

            missmatchErr_arr.append(percent_missmatch)
            pixelErr_arr.append(pixel_error)
            pixelPercentErr_arr.append(pixel_error_percents)

            predictedCoord_arr.append(predicted_point)
            anotatedCoord_arr.append(test_point)
            eucledian_distances_all.append(eucledian_distance)
            
            dictionary.update({
                        landmark_names[idx]:{
                            "Igor point coordinates [x,y]": test_point,
                            "Andrej point coordinates [x,y]": predicted_point,
                            "Percentage missmatch [x,y]": percent_missmatch,
                            "Pixel error [x,y]": pixel_error,
                            "mm error [x,y]": [pixel_error[0] / 3.6, pixel_error[1] / 3.6],
                            "Percent pixel error [x,y]": pixel_error_percents,
                            "Eucledian distance pixel": eucledian_distance,
                            "Eucledian distance mm": eucledian_distance / 3.6,
                            },
            })

        # Save JSON file with data
        name = yolov8_functions.filename_creation(path, ".json")
        filename = save_path + "/" + name
        yolov8_functions.create_json_datafile(dictionary, filename)

        # open image based on point name
        yolov8_functions.save_evaluation_image(image, filename, points_i, points_a)
        image_shape = np.array(image).shape # x and y are switched
        square_side = image_shape[0]*square_size_ratio
        half_side = math.ceil(square_side/2)
        
        sorted_names = ["FHC", "FNOC", "TKC", "TML", "sFMDA1", "sFMDA2", "sTMA1", "sTMA2"]
        for i, p in enumerate(points_a):
            coor_y = 1
            coor_x = 0
            pp = points_i[i]
            percent_y = yolov8_functions.percentage(pp[coor_y], p[coor_y])
            percent_x = yolov8_functions.percentage(pp[coor_x], p[coor_x])

            p = np.array(p)
            im = np.array(image)[round(p[1]-half_side):round(p[1]+half_side),round(p[0]-half_side):round(p[0]+half_side)]

            fig, ax = plt.subplots()
            ax.plot(half_side, half_side, marker='x', color="black")  # test point
            ax.plot(half_side + (p[0] - pp[0]),half_side + (p[1] - pp[1]), marker='+', color="red")  # predicted point

            plt.imshow(im, cmap="gray")
            plt.savefig(filename + "_" + sorted_names[i] + '.png')
            #plt.show()
            plt.close()

if (len(predictedCoord_arr) != 0):
    # Error statistics - explanation in -/documents/graphs_explanation.txt
    for i in pixelErr_arr:
        x = i[coor_x]/ map_factor
        y = i[coor_y]/ map_factor
        p = [x,y]
        mmmErr_arr.append(p)

    for i in eucledian_distances_all:
        x = i / map_factor
        eucledian_distances_all_mm.append(x)

    # Eucledian distances per point
    t_points = [fhc_points_t, 
                aF1_points_t, 
                fnoc_points_t, 
                tkc_points_t, 
                sfdma1_points_t, 
                sfdma2_points_t, 
                stma1_points_t, 
                stma2_points_t, 
                tml_points_t]
    
    p_points = [fhc_points_p, 
                aF1_points_p, 
                fnoc_points_p, 
                tkc_points_p, 
                sfdma1_points_p, 
                sfdma2_points_p, 
                stma1_points_p, 
                stma2_points_p, 
                tml_points_p]
    
    eucledian_fhc = []
    eucledian_aF1 = []
    eucledian_fnoc = []
    eucledian_tkc = []
    eucledian_sfmda1 = []
    eucledian_sfmda2 = []
    eucledian_stma1 = []
    eucledian_stma2 = []
    eucledian_tml = []

    eu_distances = [eucledian_fhc,
                    eucledian_aF1,
                    eucledian_fnoc,
                    eucledian_tkc,
                    eucledian_sfmda1,
                    eucledian_sfmda2,
                    eucledian_stma1,
                    eucledian_stma2,
                    eucledian_tml] 
    
    eucledian_fhc_mm = []
    eucledian_aF1_mm = []
    eucledian_fnoc_mm = []
    eucledian_tkc_mm = []
    eucledian_sfmda1_mm = []
    eucledian_sfmda2_mm = []
    eucledian_stma1_mm = []
    eucledian_stma2_mm = []
    eucledian_tml_mm = []
    
    eu_distances_mm = [eucledian_fhc_mm,
                    eucledian_aF1_mm,
                    eucledian_fnoc_mm,
                    eucledian_tkc_mm,
                    eucledian_sfmda1_mm,
                    eucledian_sfmda2_mm,
                    eucledian_stma1_mm,
                    eucledian_stma2_mm,
                    eucledian_tml_mm] 

 
    for j,arr in enumerate(p_points):
        for i, value in enumerate(arr):
            eucledian_distance = yolov8_functions.euclidean_distance(p_points[j][i], t_points[j][i])
            eu_distances[j].append(eucledian_distance)

    
    for i, arr in enumerate(eu_distances):
        for j, value in enumerate(arr):
            x = value / map_factor
            eu_distances_mm[i].append(x)

    dictionary = {
        "Average pixel error [x,y]": yolov8_functions.get_average(pixelErr_arr),
        "Average mm error [x,y]": yolov8_functions.get_average(mmmErr_arr),
        "Average euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_distances_all), yolov8_functions.get_average_one(eucledian_distances_all_mm)],
        "Average FHC error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(fhc_points_p, fhc_points_t),
        "Average FHC euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_fhc), yolov8_functions.get_average_one(eucledian_fhc_mm)],
        "Average FNOC error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(fnoc_points_p, fnoc_points_t),
        "Average FNOC euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_fnoc), yolov8_functions.get_average_one(eucledian_fnoc_mm)],
        "Average TKC error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(tkc_points_p, tkc_points_t),
        "Average TKC euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_tkc), yolov8_functions.get_average_one(eucledian_tkc_mm)],
        "Average sFDMA1 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(sfdma1_points_p, sfdma1_points_t),
        "Average sFDMA1 euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_sfmda1), yolov8_functions.get_average_one(eucledian_sfmda1_mm)],
        "Average sFDMA2 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(sfdma2_points_p, sfdma2_points_t),
        "Average sFDMA2 euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_sfmda2), yolov8_functions.get_average_one(eucledian_sfmda2_mm)],
        "Average sTMA1 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(stma1_points_p, stma1_points_t),
        "Average sTMA1 euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_stma1), yolov8_functions.get_average_one(eucledian_stma1_mm)],
        "Average sTMA2 error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(stma2_points_p, stma2_points_t),
        "Average sTMA2 euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_stma2), yolov8_functions.get_average_one(eucledian_stma2_mm)],
        "Average TML error [[x,y]pixel, [x,y]mm]": yolov8_functions.get_average_points(tml_points_p, tml_points_t),
        "Average TML euclidean distance [pixel, mm]": [yolov8_functions.get_average_one(eucledian_tml), yolov8_functions.get_average_one(eucledian_tml_mm)],
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

    # Points plots
    test_arrs = [fhc_points_t, fnoc_points_t, tkc_points_t, sfdma1_points_t, sfdma2_points_t, stma1_points_t, stma2_points_t, tml_points_t]
    predicted_arrs = [fhc_points_p, fnoc_points_p, tkc_points_p, sfdma1_points_p, sfdma2_points_p, stma1_points_p, stma2_points_p, tml_points_p]
    for idx, name in enumerate(landmark_names_no_aF1):

        test_data_x, test_data_y = yolov8_functions.extract_points(test_arrs[idx])
        predicted_data_x, predicted_data_y = yolov8_functions.extract_points(predicted_arrs[idx])
        yolov8_functions.box_plot(abs(test_data_x - predicted_data_x), name + " X", statistics_path)
        yolov8_functions.box_plot(abs(test_data_y - predicted_data_y), name + " Y", statistics_path)