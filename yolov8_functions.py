import numpy as np
import json
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image
import pathlib
import math

#### Functions:

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
    bc = (p1[0]**2 + p1[1]**2- temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
   
    if abs(det) < 1.0e-6:
        return (None, np.inf)
   
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

    return ((cx, cy), radius)

def get_points(json_file_path, faktor_preslikave):
  
    # Opening JSON file
    f = open(json_file_path)
    data = json.load(f)
    f.close()

    # Get control points data from json
    controlPoints = data['markups'][0]['controlPoints']
    controlPointsArray = []

    # Iterating through control points
    for i in controlPoints:
        controlPointsArray.append(i['position'])

    ### Get data from .json files
    if len(controlPointsArray) > 1:
        p = np.abs(controlPointsArray[0])
        p1_x = p[0]
        p1_z = p[2]
        p = np.abs(controlPointsArray[1])
        p2_x = p[0]
        p2_z = p[2]
        p = np.abs(controlPointsArray[2])
        p3_x = p[0]
        p3_z = p[2]

        ### Get 3 points cirlce center and radius
        center, radius = calc_circle_center((p1_x,p1_z), (p2_x,p2_z), (p3_x,p3_z))

        # faktor za translacijo med RAS/LPS v voxels
        center = (center[0] * faktor_preslikave, center[1] * faktor_preslikave)

        # zaokroži na celo število
        points = [round(center[0]), round(center[1])]
        #print("FHC enter coordinates: " + str(center))

    else:
        p = np.abs(controlPointsArray[0])
        p1_x = p[0]
        p1_z = p[2]

        # faktor za translacijo med RAS/LPS v voxels
        points = (p1_x * faktor_preslikave, p1_z * faktor_preslikave)

        # zaokroži na celo število
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

    plt.imshow(img)
    plt.savefig(filename + '.png')  # save image&markers to png
    plt.cla()
    plt.clf()
    plt.close()

def filename_creation(path, name, word):
    return name.replace(path+word+'/', "")

def create_json_datafile(dict, name, p_name=""):
    # Serializing json
    json_object = json.dumps(dict, indent=4)

    # Writing to sample.json
    if (p_name == ""):
        with open(name + '.json', "w") as outfile:
            outfile.write(json_object)
    else: 
        with open(name + "_" + p_name + '.json', "w") as outfile:
            outfile.write(json_object)

def get_zoomed_image_part(image_shape, square_size_ratio, point, img, filename):
    square_side = image_shape[0]*square_size_ratio  # define square side size
    square_image = img[int(point[1]-square_side/2):int(point[1]+square_side/2),int(point[0]-square_side/2):int(point[0]+square_side/2)]
    center = [dim // 2 for dim in square_image.shape]
    #center = square_image.shape
    #center = [int(center[0]/2),int(center[1]/2)]
    fig, ax = plt.subplots()
    ax.plot(*center, marker='.', color="white")
    plt.imshow(square_image)
    plt.savefig(filename + '.png')
    plt.close()
    return square_image

def create_landmarks_file(point, img_shape, sqr, rat, filename, more_points ,point_name=""):
    idx = 0
    data = []

    # preveri za velikost slike v obeh smereh !!!!
    if more_points == 0:

        # izračunaj procent sirine/visine na sliki za točko
        x_percent, y_percent = get_coordinate_percent(point, img_shape)

        # alert in case of error
        if x_percent >= 1:
            print("Error: x_percent >= 1")
        if y_percent >= 1:
            print("Error: y_percent >= 1")
        
        # dodaj class
        data.append(idx)
        # dodaj vrednosti za točko in kvadrat
        # square center X, X
        data.append(x_percent)
        data.append(y_percent)
        # width
        data.append(sqr*rat)
        # heigth
        data.append(sqr)

        # landmark X, Y
        data.append(x_percent)
        data.append(y_percent)
        # visibility of point
        data.append(2)

        # add next row
        data.append('\n')

    else:
        for p in point:

            # izračunaj procent sirine/visine na sliki za točko
            x_percent, y_percent = get_coordinate_percent(p, img_shape)

            # alert in case of error
            if x_percent >= 1:
                print("Error: x_percent >= 1")
            if y_percent >= 1:
                print("Error: y_percent >= 1")

             # dodaj class
            data.append(idx)
            # dodaj vrednosti za točko in kvadrat
            # square center X, X
            data.append(x_percent)
            data.append(y_percent)
            # width
            data.append(sqr*rat)
            # heigth
            data.append(sqr)
            # landmark X, Y
            data.append(x_percent)
            data.append(y_percent)
            # visibility of point
            data.append(2)
            # add next row
            data.append('\n')

            idx += 1

    # saving the points to txt
    if (point_name == ""):
        with open(filename + '.txt', "w") as f:
            for w in data:
                f.write(str(w) + " ")
    else:
        with open(filename + '_' + point_name + '.txt', "w") as f:
            for w in data:
                f.write(str(w) + " ")

def get_coordinate_percent(point, img_size):
    return point[0] / img_size[1], point[1] / img_size[0]

def main_func(sav_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data):

        # Shrani sliko v JPG
        filename = sav_path + "/ALL/images/" + data + "/" + name
        matplotlib.image.imsave(filename + '.jpg', data_arr)

        # kreiraj JSON dataset
        dictionary = {
            "Image name": filename,
            "Point names": point_names,
            "Point coordinates": points,
            "Image_size": orig_image_shape,
        }

        filename = sav_path + "/JSON/" + name
        create_json_datafile(dictionary, filename)

        # kreiraj txt zapis za točke
        filename = sav_path + "/ALL/labels/" + data + "/" + name
        create_landmarks_file(points, orig_image_shape, square, orig_img_ratio, filename, 1)

        # izreži kvadrate na sliki za kaskadne predikcije
        i = 0
        for p in points:

            # shrani sliko v PNG
            filename = sav_path + "/PNGs/" + name + "_" + point_names[i]
            #img = get_zoomed_image_part(orig_image_shape, square, p, data_arr, filename)

            # slice image - remove img zgoraj!
            img, p_changed, changed_image_shape, changed_img_ratio = slice_image_3_parts(orig_image_shape, square, p, data_arr, point_names[i], filename)

            # Shrani sliko v JPG
            filename = sav_path + "/" + point_names[i] + "/images/"  + data + "/" + name + "_" + point_names[i]
            matplotlib.image.imsave(filename + '.jpg', img)

            # kreiraj JSON dataset
            # x in y koordinati v points sta obrnjeni
            dictionary = {
                "Image name": filename,
                "Point name": point_names[i],
                "Point coordinates": p,
                "Changed coordinates": p_changed,
                "Image_size": orig_image_shape,
                "Zoomed_image_size": img.shape
            }

            filename = sav_path + "/JSON/" + name + "_" + point_names[i]
            create_json_datafile(dictionary, filename)

            # kreiraj txt zapis za točke
            filename = sav_path + "/" + point_names[i] + "/labels/"  + data + "/" + name
            # square je tukaj večji, ker je na sliki za učenje potem večji, 0.1 je premajhen
            square = 0.2
            create_landmarks_file(p_changed, changed_image_shape, square, changed_img_ratio, filename, 0, point_names[i])
            square = 0.1

            i += 1

def get_dirs(path):
    return [str(item) for item in pathlib.Path(path).iterdir() if ".DS_Store" not in str(item)]

def get_nrrd_paths(dirs, workingDirPath):
    paths = []
    for d in dirs:
        d = workingDirPath + str(d) + '/'
        d = pathlib.Path(d)
        for item in d.iterdir():
            i = str(item)
            if (i.find(".nrrd") != -1):
                paths.append(i)
    
    return paths

def get_json_paths(dirs, point_names):
    paths = []
    for d in dirs:
        d = pathlib.Path(d)
        for item in d.iterdir():
            i = str(item)
            # ali je smiselno, da čekiram katere točke so na voljo?
            if (str(i).find(".json") != -1):
                for name in point_names:
                    if(str(i).find(name) != -1):
                        paths.append(str(i))
    
    return paths

def get_jpg_paths(dirs, point_names, path, all_imgs):
    paths = []
    for d in dirs:
        name = d.replace(path, "")
        if name in point_names:
            d = pathlib.Path(d+all_imgs)

            for item in d.iterdir():
                i = str(item)
                if (str(i).find(".jpg") != -1):
                    paths.append(str(i))

    return paths

def get_paths_word(word, list_of_paths):
    return [path for path in list_of_paths if word in path]

def full_image_save_predict(points, img, filename):
    fig, ax = plt.subplots()

    # plot points [FHC, TKC, TML, aF1]
    for point in points: 
        ax.plot(*point, marker='.', color="white")

    plt.imshow(img)
    plt.savefig(filename + '.png')
    plt.close()

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

    plt.imshow(image_part)
    plt.savefig(filename + '.png')
    plt.close()
    
    return image_part, point, image_part.shape, (image_part.shape[0] / image_part.shape[1])
