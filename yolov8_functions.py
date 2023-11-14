""" Functions library """
import numpy as np
import json
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image
import pathlib
import math
import os
import shutil
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns

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

def get_sPoints(paths, scale_factor):
    points = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        control_points = [i['position'] for i in data['markups'][0]['controlPoints']]

        p = np.abs(control_points[0])
        p1_x = p[0]
        p1_z = p[2]
        p = np.abs(control_points[1])
        p2_x = p[0]
        p2_z = p[2]

        point = [[round(p1_x*scale_factor), round(p1_z*scale_factor)],[round(p2_x*scale_factor), round(p2_z*scale_factor)]]
        points.append(point)
    return points

def get_points(json_file_path, scale_factor):
    with open(json_file_path) as f:
        data = json.load(f)

    control_points = [i['position'] for i in data['markups'][0]['controlPoints']]

    if len(control_points) > 1:
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

    else:
        p = np.abs(control_points[0])
        p1_x = p[0]
        p1_z = p[2]

        # faktor za translacijo med RAS/LPS v voxels
        points = (p1_x * scale_factor, p1_z * scale_factor)
        points = [round(points[0]), round(points[1])]

    return points

def create_point_array(paths, scale_factor):
    return [get_points(path, scale_factor) for path in paths]

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

def main_func(save_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data, s_points, s_points_names):

    # save image to JPG
    filename = f"{save_path}/ALL/images/{data}/{name}"
    plt.title(filename)
    matplotlib.image.imsave(f"{filename}.jpg", data_arr, cmap="gray")

    dictionary = {
        "Image name": filename,
        "Point names": point_names,
        "Point coordinates": points,
        "sTMA, sFMDA coordinates": s_points,
        "Image_size": orig_image_shape,
    }

    create_json_datafile(dictionary, f"{save_path}/JSON/{name}")
    create_landmarks_file(points, orig_image_shape, square, orig_img_ratio, f"{save_path}/ALL/labels/{data}/{name}")

    # dodaj še s_points v ločen landmark file + ločen save_path na foro spodnjega
    # delal bo tko kokr za tkc/fnoc
    # obe točke za sfdma in stma
    for i, point_arr in enumerate(s_points):
        img, p_changed, changed_image_shape, changed_img_ratio = sPoints_imageParts(orig_image_shape, square, point_arr, data_arr, f"{save_path}/PNGs/{name}_{s_points_names[i]}")
        
        filename = f"{save_path}/{s_points_names[i]}/images/{data}/{name}_{s_points_names[i]}"
        matplotlib.image.imsave(f"{filename}.jpg", img, cmap="gray")

        dictionary = {
            "Image name": filename,
            "Point name": s_points_names[i],
            "Point coordinates": point_arr,
            "Changed coordinates": p_changed,
            "Image_size": orig_image_shape,
            "Zoomed_image_size": img.shape
        }

        # popravi, da imaš obe točki zapisnai v coco dateset + json
        create_json_datafile(dictionary, f"{save_path}/JSON/{name}_{s_points_names[i]}")
        create_landmarks_file(p_changed, changed_image_shape, 0.2, changed_img_ratio, f"{save_path}/{s_points_names[i]}/labels/{data}/{name}", s_points_names[i])


    # get smaller pictures of landmarks for cascade learning
    for i, point in enumerate(points):

        img, p_changed, changed_image_shape, changed_img_ratio = slice_image_3_parts(orig_image_shape, square, point, data_arr, point_names[i], f"{save_path}/PNGs/{name}_{point_names[i]}")

        filename = f"{save_path}/{point_names[i]}/images/{data}/{name}_{point_names[i]}"
        matplotlib.image.imsave(f"{filename}.jpg", img, cmap="gray")

        dictionary = {
            "Image name": filename,
            "Point name": point_names[i],
            "Point coordinates": point, #x&y coordinates are reversed
            "Changed coordinates": p_changed,
            "Image_size": orig_image_shape,
            "Zoomed_image_size": img.shape
        }

        create_json_datafile(dictionary, f"{save_path}/JSON/{name}_{point_names[i]}")
        create_landmarks_file(p_changed, changed_image_shape, 0.2, changed_img_ratio, f"{save_path}/{point_names[i]}/labels/{data}/{name}", point_names[i])

def get_dirs(path):
    return [str(item) for item in pathlib.Path(path).iterdir() if ".DS_Store" not in str(item)]

def get_nrrd_paths(dirs, working_dir_path):
    return [str(item) for d in dirs for item in pathlib.Path(f"{working_dir_path}{d}/").iterdir() if ".nrrd" in str(item)]

def get_json_paths(dirs, point_names):
    return [str(item) for d in dirs for item in pathlib.Path(d).iterdir() if ".json" in str(item) and any(name in str(item) for name in point_names)]

def get_jpg_paths(directory):
    return [str(item) for item in pathlib.Path(directory).iterdir() if ".jpg" in str(item)]

def sPoints_imageParts(image_shape, square, points, img, filename):
    n_points = []
    square_side = image_shape[0]*square # define square side size
    height = image_shape[0]
    two_thirds = math.ceil(height * 2 / 3)
    one_third = math.ceil(height / 3)

    image_part = img[one_third:two_thirds,:]

    # new point coordinates
    for point in points: 
        point = [point[0], point[1] - one_third]
        n_points.append(point)

    fig, ax = plt.subplots()
    for point in n_points:
        ax.plot(*point, marker='.', color="white")
        rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    plt.imshow(image_part, cmap="gray")
    plt.title(filename)
    plt.savefig(filename + '.png')
    plt.close()

    return image_part, n_points, image_part.shape, (image_part.shape[0] / image_part.shape[1])

def slice_image_3_parts(image_shape, square, point, img, point_name, filename):
    square_side = image_shape[0]*square # define square side size
    height = image_shape[0]
    two_thirds = math.ceil(height * 2 / 3)
    one_third = math.ceil(height / 3)

    # točka na sliki + izrez kvadrata iz slike
    if point_name == 'FHC':
        image_part = img[0:one_third,:]
    elif point_name in ['TKC','FNOC']:
        image_part = img[one_third:two_thirds,:]
        point = [point[0], point[1] - one_third]
    elif point_name == 'TML':
        image_part = img[two_thirds:height,:]
        point = [point[0], point[1] - two_thirds]
    elif point_name == 'aF1':
        image_part = img[0:math.ceil(height / 2),:]
    
    fig, ax = plt.subplots()
    ax.plot(*point, marker='.', color="white")
    rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
    ax.add_patch(rect)

    plt.imshow(image_part, cmap="gray")
    plt.title(filename)
    plt.savefig(filename + '.png')
    plt.close()
    
    return image_part, point, image_part.shape, (image_part.shape[0] / image_part.shape[1])

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
    
    err_avg_x = sum(err[0] for err in err_arr) / len(err_arr)
    err_avg_y = sum(err[1] for err in err_arr) / len(err_arr)
    
    return err_avg_x, err_avg_y

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
