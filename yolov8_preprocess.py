""" Prepare pictures and data for YoloV8 training. """
import yolov8_functions

# Dataset path:
workingDirPath = "/Users/jankovic/Documents/yolov8/"
path = "./data/RTG_dataset/"
sav_path = './data/dataset'
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1']
filter_val = 10000
faktor_preslikave = 3.6
square = 0.1    # square size
num_parts = 3   # number of parts on image

# nrrd files paths
directories = yolov8_functions.get_dirs(path)
nrrd_image_paths = yolov8_functions.get_nrrd_paths(directories, workingDirPath)
point_json_paths = yolov8_functions.get_json_paths(directories, point_names)

# Test = 20%, Train = 80%, Validate = 20% of Train
train, test, val = yolov8_functions.split_train_test_val_data(nrrd_image_paths)

# create dataset archive
yolov8_functions.dataset_archive(sav_path)

# script 
j = 0
u = len(point_names)
# n = nrrd image path
for n in nrrd_image_paths:   
    
    # paths to points for single image
    p_paths = []    
    for i in range(u):
        i = i + (j*u)
        p_paths.append(point_json_paths[i])
    j += 1

    # original image & points
    data_arr, orig_image_shape, orig_img_ratio = yolov8_functions.preprocess_image(n, filter_val)
    points = yolov8_functions.create_point_array(p_paths, faktor_preslikave)
    name = yolov8_functions.filename_creation(path, n, ".nrrd")
    filename = sav_path + "/PNGs/" + name
    yolov8_functions.save_image(orig_image_shape, square, points, data_arr, filename)

    # split data
    if(n in train):
        data = "train"
    elif(n in val):
        data = "val"
    else:
        data = "test"

    yolov8_functions.main_func(sav_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data)

print("Script is done!")