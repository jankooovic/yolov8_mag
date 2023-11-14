""" Prepare pictures and data for YoloV8 training. """
import yolov8_functions
import math

# Dataset path:
workingDirPath = "./"
path = "./data/RTG_dataset/"
save_path = './data/dataset'
point_names_all = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1', 'sFMDA', 'sTMA']
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1']
sPoints_SavePath = './data/dataset/Points/'
filter_val = 10000
map_factor = 3.6
square = 0.1    # square size
num_parts = 3   # number of parts on image

# nrrd files paths
directories = yolov8_functions.get_dirs(path)
nrrd_image_paths = yolov8_functions.get_nrrd_paths(directories, workingDirPath)
point_json_paths = yolov8_functions.get_json_paths(directories, point_names_all)

# Test = 20%, Train = 80%, Validate = 20% of Train
train, test, val = yolov8_functions.split_train_test_val_data(nrrd_image_paths)

# create dataset archive
yolov8_functions.dataset_archive(save_path)

# script 
j = 0
u = len(point_names_all)
# n = nrrd image path
for n in nrrd_image_paths:   
    
    # paths to points for single image
    p_paths = []   
    for i in range(u):
        i = i + (j*u)
        p_paths.append(point_json_paths[i])
    j += 1

    #remove sfmda and stma paths
    s_paths = []
    for idx, path in enumerate(p_paths):
        if ("sTMA" in path or "sFMDA" in path):
            p_paths.pop(idx)
            s_paths.append(path)

    # original image & points
    data_arr, orig_image_shape, orig_img_ratio = yolov8_functions.preprocess_image(n, filter_val)
    points = yolov8_functions.create_point_array(p_paths, map_factor)
    name = yolov8_functions.filename_creation(n, ".nrrd")
    filename = save_path + "/PNGs/" + name
    yolov8_functions.save_image(orig_image_shape, square, points, data_arr, filename)

    print("Processing image: " + name)

    # split data
    if(n in train):
        data = "train"
    elif(n in val):
        data = "val"
    else:
        data = "test"
    
    ### s_points slika + točke, da se vidijo - all_paths???
    s_points = yolov8_functions.get_sPoints(s_paths, map_factor)

    yolov8_functions.main_func(save_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data, s_points)

print("Script is done!")