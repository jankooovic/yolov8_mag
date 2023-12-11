""" Functions library """
import numpy as np
import json
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image
import pathlib
import os
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns
import math

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def open_nrrd_image(file_path):
    data, _ = nrrd.read(file_path)
    return np.array(data)

def rotate_array(array, nm):
    # nm = number of 90 degree rotations clockwise
    return np.rot90(array, nm, axes=(1, 0))

def preprocess_image(file_path, filter_val):
    array = open_nrrd_image(file_path)
    array = array[:,:,0]        # Get only picture l x w
    array[array > filter_val] = 0   # filter data
    normalized_array = normalize_data(array)
    rotated_array = rotate_array(normalized_array, 1)
    flipped_array = np.fliplr(rotated_array)
    img_shape = flipped_array.shape
    img_ratio = img_shape[0] / img_shape[1] # image heigth / width ratio
    return flipped_array, img_shape, img_ratio

def calc_circle_center(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
   
    if abs(det) < 1.0e-6:
        return (None, np.inf)
   
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

    return (cx, cy), radius

def get_points(json_file_path, scale_factor):
    with open(json_file_path) as f:
        data = json.load(f)

    control_points = [i['position'] for i in data['markups'][0]['controlPoints']]

    if len(control_points) == 3:
        p = np.abs(control_points[0])
        p1_x = p[0]
        p1_z = p[2]
        p = np.abs(control_points[1])
        p2_x = p[0]
        p2_z = p[2]
        p = np.abs(control_points[2])
        p3_x = p[0]
        p3_z = p[2]

        ### Get 3 points cirlce center and radius
        center, radius = calc_circle_center((p1_x,p1_z), (p2_x,p2_z), (p3_x,p3_z))

        # faktor za translacijo med RAS/LPS v voxels
        center = (center[0] * scale_factor, center[1] * scale_factor)
        points = [round(center[0]), round(center[1])]
        return points, "good"
    
    elif len(control_points) == 2:
        p = np.abs(control_points[0])
        p1_x = p[0]
        p1_z = p[2]
        p = np.abs(control_points[1])
        p2_x = p[0]
        p2_z = p[2]

        # faktor za translacijo med RAS/LPS v voxels
        point1 = [round(p1_x*scale_factor), round(p1_z*scale_factor)]
        point2 = [round(p2_x*scale_factor), round(p2_z*scale_factor)]
        return [point1, point2], "sPoints"

    else:
        p = np.abs(control_points[0])
        p1_x = p[0]
        p1_z = p[2]

        # faktor za translacijo med RAS/LPS v voxels
        points = (p1_x * scale_factor, p1_z * scale_factor)
        points = [round(points[0]), round(points[1])]
        return points, "good"

def create_point_array(paths, scale_factor):
    points = []
    for path in paths:
        ps, sP = get_points(path, scale_factor)
        if sP == 'sPoints':
            points.append(ps[0])
            points.append(ps[1])
        else:
            points.append(ps)

    return points

def save_image(image_shape, square_size_ratio, points, img, filename):
    square_side = image_shape[0]*square_size_ratio  # define square side size
    fig, ax = plt.subplots()

    # plot points [FHC, TKC, TML, aF1]
    for point in points: 
        ax.plot(*point, marker='.', color="white")
        rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    plt.imshow(img, cmap="gray")
    plt.title(filename)
    plt.savefig(filename + '.png')  # save image&markers to png
    plt.cla()
    plt.clf()
    plt.close()

def filename_creation(name, word, sign="/"):
    filename = name.rsplit(sign,1)
    filename = filename[1].replace(word,"")
    return filename

def create_json_datafile(data_dict, name, prefix=""):
    json_object = json.dumps(data_dict, indent=4)
    with open(f"{name}_{prefix}.json" if prefix else f"{name}.json", "w") as outfile:
            outfile.write(json_object)

def get_zoomed_image_part(image_shape, square_size_ratio, point, img, filename):
    square_side = image_shape[0]*square_size_ratio  # define square side size
    square_image = img[int(point[1]-square_side/2):int(point[1]+square_side/2),int(point[0]-square_side/2):int(point[0]+square_side/2)]
    center = [dim // 2 for dim in square_image.shape]
    fig, ax = plt.subplots()
    ax.plot(*center, marker='.', color="white")
    plt.imshow(square_image, cmap="gray")
    plt.title(filename)
    plt.savefig(filename + '.png')
    plt.close()
    return square_image

def create_landmarks_file(points, img_shape, sqr, rat, filename, point_name=""):
    data = [] 
    n = 0
    for idx, point in enumerate(points):

        if isinstance(point, int):
            point = points

        x_percent, y_percent = get_coordinate_percent(point, img_shape) # get cooridnate width/heigth percentage

        if x_percent >= 1 or y_percent >= 1:
            print("Error: x_percent >= 1 or y_percent >= 1")
            continue

        data.extend([idx,   # class
                    x_percent, # square center X, X
                    y_percent,
                    sqr*rat,   # width
                    sqr,   # heigth
                    x_percent, # landmark X, Y
                    y_percent,
                    2, # visibility of point
                    '\n']) # add next row
        
        # end this loop if this is only one point
        if (point_name == 'sTMA'):
            n = 1
        elif (point_name == 'sFMDA'):
            n = 1
        elif len(points) == 2:
            break
        
    # saving the points to txt
    with open(f"{filename}_{point_name}.txt" if point_name else f"{filename}.txt", "w") as f:
        f.write(" ".join(map(str, data)))

def get_coordinate_percent(point, img_size):
    return point[0] / img_size[1], point[1] / img_size[0]

def main_func(save_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data):

    # save image to JPG
    filename = f"{save_path}/ALL/images/{data}/{name}"
    plt.title(filename)
    matplotlib.image.imsave(f"{filename}.jpg", data_arr, cmap="gray")

    dictionary = {
        "Image name": filename,
        'sTMA': points[0] + points[1],
        'FHC':points[2],
        'sFMDA':points[3] + points[4],
        'TKC':points[5],
        'TML':points[6],
        'FNOC':points[7],
        'aF1':points[8],
        "Image_size": orig_image_shape,
    }

    create_json_datafile(dictionary, f"{save_path}/JSON/{name}")
    create_landmarks_file(points, orig_image_shape, square, orig_img_ratio, f"{save_path}/ALL/labels/{data}/{name}")

def get_dirs(path):
    return [str(item) for item in pathlib.Path(path).iterdir() if ".DS_Store" not in str(item)]

def get_nrrd_paths(dirs, working_dir_path):
    return [str(item) for d in dirs for item in pathlib.Path(f"{working_dir_path}{d}/").iterdir() if ".nrrd" in str(item)]

def get_json_paths(dirs, point_names):
    return [str(item) for d in dirs for item in pathlib.Path(d).iterdir() if ".json" in str(item) and any(name in str(item) for name in point_names)]

def get_jpg_paths(directory):
    return [str(item) for item in pathlib.Path(directory).iterdir() if ".jpg" in str(item)]

def dataset_archive(save_path):
    now = datetime.now()
    date_time = now.strftime("_%d-%m-%Y %H-%M")
    os.rename(save_path,save_path + date_time)
    shutil.copytree(save_path + "_template",save_path)    # copy dataset template to dataset

def split_train_test_val_data(nrrd_image_paths):
    train,test=train_test_split(nrrd_image_paths,test_size=0.2) # Train/Test split 80/20
    train,val=train_test_split(train,test_size=0.2) # Train/Val split 80/20
    return train, test, val

def save_prediction_image(points, img, filename):
    fig, ax = plt.subplots()
    # plot points [FHC, TKC, TML, aF1]
    for point in points: 
        ax.plot(*point, marker='.', color="white")
    plt.imshow(img, cmap="gray")
    plt.title(filename)
    plt.savefig(filename + '.png')
    plt.close()

def percentage(part, whole):
    return (float(part)/float(whole) * 100)

def save_evaluation_image(image, filename, test_coordinates, predicted_coordinates):

    fig, ax = plt.subplots()
    # plot reference points
    for point in test_coordinates: 
        ax.plot(*point, marker='o', color="white")
    
    # plot predicted points
    for point in predicted_coordinates: 
        ax.plot(*point, marker='+', color="red")  # naredi, da imajo točke druge abrve

    plt.imshow(image, cmap="gray")
    plt.title(filename)
    # #plt.show()
    plt.savefig(filename + '.png')
    plt.cla()
    plt.clf()
    plt.close()
    return image

def open_image(path):
    return Image.open(path).convert("L")

def get_average(err_arr):
    if not err_arr:
        return 0, 0  # Handle empty array to avoid division by zero
    
    x_err = []
    y_err = []
    for err in err_arr:  
        x_err.append(err[0])
        y_err.append(err[1])
    
    err_avg_x = sum(x_err) / len(err_arr)
    err_avg_y = sum(y_err) / len(err_arr)
    
    return err_avg_x, err_avg_y

def get_average_one(err_arr):
    if not err_arr:
        return 0  # Handle empty array to avoid division by zero
 
    err_avg_x = sum(err_arr) / len(err_arr)
    
    return err_avg_x

def get_average_points(test_ps, predicted_ps):

    error_x_pixel = []
    error_y_pixel = []
    error_x_mm = []
    error_y_mm = []

    for i, p in enumerate(test_ps):
        x_t = test_ps[i][0]
        y_t = test_ps[i][1]
        x_p = predicted_ps[i][0]
        y_p = predicted_ps[i][1]

        err_x_pix = abs(x_t - x_p)
        err_y_pix = abs(y_t - y_p)
        err_x_mm = err_x_pix / 3.6
        err_y_mm = err_y_pix / 3.6
        
        error_x_pixel.append(err_x_pix)
        error_y_pixel.append(err_y_pix)
        error_x_mm.append(err_x_mm)
        error_y_mm.append(err_y_mm)

        err_avg_x_pixel = sum(error_x_pixel) / len(error_x_pixel)
        err_avg_y_pixel = sum(error_y_pixel) / len(error_y_pixel)
        err_avg_x_mm = sum(error_x_mm) / len(error_x_mm)
        err_avg_y_mm = sum(error_y_mm) / len(error_y_mm)
    
    return [[err_avg_x_pixel, err_avg_y_pixel],[err_avg_x_mm, err_avg_y_mm]]

def scatter_plot(measured_coordinates, true_coordinates, coordinate, sav_path):
    name = 'Scatter Plot of Measured vs. True Coordinates for ' + coordinate
    plt.scatter(measured_coordinates, true_coordinates, s=20, alpha=0.7)
    plt.xlabel('Measured Coordinates')
    plt.ylabel('True Coordinates')
    plt.title(name)
    plt.savefig(sav_path + "/" + name + '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def residual_plot(measured_coordinates, true_coordinates, coordinate, sav_path):
    name = 'Residual Plot for ' + coordinate
    residuals = measured_coordinates - true_coordinates
    plt.scatter(measured_coordinates, residuals, s=20, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Measured Coordinates')
    plt.ylabel('Residuals')
    plt.title(name)
    plt.savefig(sav_path + "/" + name+ '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def histogram_of_errors(errors, coordinate, sav_path):
    name = 'Histogram of Errors for ' + coordinate
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Errors')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.savefig(sav_path + "/" + name+ '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def qq_plot(errors, coordinate, sav_path):
    name = 'Q-Q Plot for ' + coordinate
    from scipy.stats import probplot
    probplot(errors, plot=plt)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(name)
    plt.savefig(sav_path + "/" + name + '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def bland_altman_plot(measured_coordinates, true_coordinates, coordinate, sav_path):
    name = 'Bland-Altman Plot for ' + coordinate
    differences = measured_coordinates - true_coordinates
    averages = (measured_coordinates + true_coordinates) / 2
    plt.scatter(averages, differences, s=20, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Averages of Measured and True Coordinates')
    plt.ylabel('Differences (Measured - True)')
    plt.title(name)
    plt.savefig(sav_path + "/" + name+ '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def box_plot(errors, coordinate, sav_path):
    name = 'Box Plot of Errors for ' + coordinate
    plt.boxplot(errors)
    plt.xlabel('Error')
    plt.ylabel('Distribution')
    plt.title(name)
    plt.savefig(sav_path + "/" + name+ '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def heatmap(measured_coordinates, true_coordinates, coordinate, sav_path):
    name = 'Error Heatmap for ' + coordinate
    plt.hist2d(measured_coordinates, true_coordinates, bins=30, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Measured Coordinates')
    plt.ylabel('True Coordinates')
    plt.title(name)
    plt.savefig(sav_path + "/" + name+ '.png')
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()

def extract_points(array):
    x_vals, y_vals = zip(*array)
    return np.array(x_vals),np.array(y_vals)

def range_and_iqr_of_differences(x_coordinates, y_coordinates):
    differences = y_coordinates - x_coordinates
    diff_range = np.ptp(differences)
    diff_iqr = np.percentile(differences, 75) - np.percentile(differences, 25)
    return diff_range, diff_iqr

def standard_deviation_of_differences(x_coordinates, y_coordinates):
    differences = y_coordinates - x_coordinates
    return np.std(differences)

def coefficient_of_variation_of_differences(x_coordinates, y_coordinates):
    differences = y_coordinates - x_coordinates
    mean_difference = np.mean(differences)
    std_dev_difference = np.std(differences)
    return (std_dev_difference / mean_difference) * 100

def violin_plot_of_differences(x_coordinates, y_coordinates, coordinate, sav_path):
    name = 'Violin Plot of Differences for ' + coordinate
    differences = y_coordinates - x_coordinates
    sns.violinplot(differences)
    plt.ylabel('Differences (True - Measured)')
    plt.title(name)
    plt.savefig(sav_path + "/" + name+ '.png')
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def sort_sPoints(test_sPoints):
    # sfmda & stma + fnoc & tkc
    s = [test_sPoints[0],test_sPoints[1],test_sPoints[4],test_sPoints[5]]
    f = [test_sPoints[2], test_sPoints[3]]

    # sort sFMDA & sTMA
    if s[0][1] > s[1][1]:
        tmp = s[0]
        s[0] = s[1]
        s[1] = tmp
    if s[2][1] > s[3][1]:
        tmp = s[2]
        s[2] = s[3]
        s[3] = tmp

    # sort FNOC & TKC
    if f[0][1] > f[1][1]:
        tmp = f[0]
        f[0] = f[1]
        f[1] = tmp
        
    sPoints = f + s

    return sPoints

def get_indices(element, lst):
    indices = []
    for i in range(len(lst)):
        if lst[i] == element:
            indices.append(i)
    return indices

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1: List or tuple representing the coordinates of the first point [x1, y1].
    - point2: List or tuple representing the coordinates of the second point [x2, y2].

    Returns:
    - The Euclidean distance between the two points.
    """
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Each point should be represented as [x, y]")

    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
