import numpy as np
import json
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image
import pathlib
import math

#### Functions:

def normalize_data_to_interval(data):
    """
    data = podatki za normalizacijo
    uporabljeno v preprocess image
    """
    data_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    return data_norm

def open_nrrd_image(nrrd_file_path):
    """
    nrrd_file_path = pot do nrrd datoteke
    uporabljeno v preprocess image
    """
    # Read data from nrrd file
    data, header = nrrd.read(nrrd_file_path)

    # Create numpy array
    data_array = np.array(data)

    return data_array

def array_rotation(array, nm):
    """
    Uporabljeno v funkciji preprocees_image
    """
    # aray = image array -> 2 dimensions
    # x = number of 90 degree rotations clockwise
    rot_array = array
    rot_array = np.rot90(rot_array, nm, axes=(1, 0))
    return rot_array

def preprocess_image(nrrd_file, filter_val):
    """
    nrrd_file = pot do nrrd datoteke
    filter_val = vrednosti na sliki, ki jih ne potrebujem
    rezultat funkcije je slika, ki je pripravljena na obdelavo + podatki o sliki
    """

    # Get nrrd image data array
    data_array = open_nrrd_image(nrrd_file)

    # Get only picture l x w
    data_array = data_array[:,:,0]

    # remove data that is not needed - filter
    data_array[data_array > filter_val] = 0

    # normalize data to interval [0,1]
    data_arary_norm = normalize_data_to_interval(data_array)

    # transform to numpy array -> to je potrebno?
    data_arary_norm_numpy = np.array(data_arary_norm)

    # rotacija slike za 90 stopinj clockwise
    data_arary_norm_numpy = array_rotation(data_arary_norm_numpy, 1)

    # flip slike 
    data_arary_norm_numpy = np.fliplr(data_arary_norm_numpy)

    # image shape[height, width]
    img_shape = data_arary_norm_numpy.shape

    # image heigth / width ratio
    img_ratio = (img_shape[0] / img_shape[1])

    return data_arary_norm_numpy, img_shape, img_ratio

def calc_circle_center(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    Uporabljeno v funckiji get_points
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
   
    if abs(det) < 1.0e-6:
        return (None, np.inf)
   
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
   
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def get_points(json_file_path, faktor_preslikave):
    """
    uporabljeno v funckiji: create_points_array
    """

    # Opening JSON file
    f = open(json_file_path)
    
    # returns JSON object as a dictionary
    data = json.load(f)

    # Closing file
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

def create_point_array(p_paths, faktor_preslikave):
    points = []
    for point_path in p_paths:
        point = get_points(point_path, faktor_preslikave)
        points.append(point)
    return points

def show_save_full_image(image_shape, square, points, img, filename):
    
    temp = np.array(img)
    fig, ax = plt.subplots()

    # define square side size - prilagodi glede na velikost objekta
    square_side = image_shape[0]*square

    # [FHC, TKC, TML, aF1]
    # plot points

    for p in points: 
        ax.plot(p[0], p[1], marker='.', color="white")
        rect = patches.Rectangle((p[0]-square_side/2, p[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    #print("Point orig:",p[0], p[1])
    plt.imshow(temp[:,:])
    # save image&markers to png
    plt.savefig(filename + '.png')
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def filename_creation(path, n, word):
    img_name = n.replace(path, "")
    img_name = img_name.replace(word, "")
    img_name = img_name.replace("/", "")
    filename = img_name
    return filename

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

def get_zoomed_image_part(image_shape, square, point, img, filename):

    # define square side size - prilagodi glede na velikost objekta
    square_side = image_shape[0]*square

    # točka na sliki + izrez kvadrata iz slike
    square_image = img[int(point[1]-square_side/2):int(point[1]+square_side/2),int(point[0]-square_side/2):int(point[0]+square_side/2)]
    center = square_image.shape
    center = [int(center[0]/2),int(center[1]/2)]

    fig, ax = plt.subplots()
    ax.plot(center[0], center[1], marker='.', color="white")
    plt.imshow(square_image[:,:])
    # save image&markers to png
    plt.savefig(filename + '.png')
    #plt.show()
    plt.cla()
    plt.clf()
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
    # koorindate točke
    p_x = point[0]
    p_y = point[1]
    
    # veliksot slike
    size_y = img_size[0]
    size_x = img_size[1]

    # procent veliksoti slike
    x_percent = (p_x / size_x)
    y_percent = (p_y / size_y)
    
    return x_percent, y_percent

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
            img, p_changed= slice_image_3_parts(orig_image_shape, square, p, data_arr, point_names[i], filename)

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
            create_landmarks_file(p, orig_image_shape, square, orig_img_ratio, filename, 0, point_names[i])

            i += 1

def get_dirs(path):

    dirs = []
    d = pathlib.Path(path)
    for item in d.iterdir():
        i = str(item)
        if(i != path + ".DS_Store"):
            dirs.append(i)

    return dirs

def get_nrrd_paths(dirs):
    nrrd_image_paths = []
    for d in dirs:
        d = pathlib.Path(d)
        for item in d.iterdir():
            i = str(item)
            if (str(i).find(".nrrd") != -1):
                nrrd_image_paths.append(str(i))
    
    return nrrd_image_paths

def get_json_paths(dirs, point_names):
    point_json_paths = []
    for d in dirs:
        d = pathlib.Path(d)
        for item in d.iterdir():
            i = str(item)
            # ali je smiselno, da čekiram katere točke so na voljo?
            if (str(i).find(".json") != -1):
                for name in point_names:
                    if(str(i).find(name) != -1):
                        point_json_paths.append(str(i))
    
    return point_json_paths

def get_jpg_paths(dirs, point_names, path, all_imgs):
    image_paths = []
    for d in dirs:
        name = d.replace(path, "")
        if name in point_names:
            d = pathlib.Path(d+all_imgs)

            for item in d.iterdir():
                i = str(item)
                if (str(i).find(".jpg") != -1):
                    image_paths.append(str(i))

    return image_paths

def get_paths_word(word, list_of_paths):

    paths = []
    for path in list_of_paths:
        if word in path:
            paths.append(path)

    return paths

def full_image_save_predict(points, img, filename):
    
    temp = np.array(img)
    fig, ax = plt.subplots()

    # [FHC, TKC, TML, aF1]
    # plot points

    for p in points: 
        ax.plot(p[0], p[1], marker='.', color="white")

    #print("Point orig:",p[0], p[1])
    plt.imshow(temp[:,:])
    # save image&markers to png
    plt.savefig(filename + '.png')
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def slice_image_3_parts(image_shape, square, point, img, point_name, filename):

   # define square side size - prilagodi glede na velikost objekta
    square_side = image_shape[0]*square

    fig, ax = plt.subplots()

    height = image_shape[0]
    two_thirds = math.ceil(height*(2/3))
    one_third = math.ceil(height*(1/3))

    # točka na sliki + izrez kvadrata iz slike
    if point_name == 'FHC':
        image_part = img[0:one_third,:]
        # plot points
        ax.plot(point[0], point[1], marker='.', color="white")
        rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    elif point_name == 'TKC':
        image_part = img[one_third:two_thirds,:]
        # plot points
        point = [point[0], point[1]-one_third]
        ax.plot(point[0], point[1], marker='.', color="white")
        rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    elif point_name == 'TML':
        image_part = img[two_thirds:height,:]
        # plot points
        point = [point[0], point[1]-two_thirds]
        ax.plot(point[0], point[1], marker='.', color="white")
        rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    elif point_name == 'aF1':
        image_part = img[0:math.ceil(height/2),:]
        # plot points
        ax.plot(point[0], point[1], marker='.', color="white")
        rect = patches.Rectangle((point[0]-square_side/2, point[1]-square_side/2), square_side, square_side, linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    #print("Image part shape:", img.shape)
    #print("Point coordinates:", point)
    plt.imshow(image_part[:,:])
    # save image&markers to png
    plt.savefig(filename + '.png')
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()
    
    return image_part, point

