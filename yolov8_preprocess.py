### Imports: 
import yolov8_functions
from sklearn.model_selection import train_test_split
import os
import shutil
from datetime import date
from datetime import datetime

"""
Prepare pictures and data for YoloV8 training.
"""

# Dataset path:
workingDirPath = "/Users/jankovic/Documents/yolov8/"
path = "./data/RTG_dataset/"
sav_path = './data/dataset'
point_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1']
filter_val = 10000
faktor_preslikave = 3.6
square = 0.1    # square size
num_parts = 3   # number of parts on image

# poti do direktorijev, kjer so nrrd datoteke
directories = yolov8_functions.get_dirs(path)
nrrd_image_paths = yolov8_functions.get_nrrd_paths(directories, workingDirPath)
point_json_paths = yolov8_functions.get_json_paths(directories, point_names)

### Make train, test and validate groups ###
"""
Split:
Test = 20%
Train = 80%
Validate = 20% of Train
"""

# Size of entire Dataset
all_num = len(nrrd_image_paths)
train,test=train_test_split(nrrd_image_paths,test_size=0.2) # Train/Test split 80/20
train,val=train_test_split(train,test_size=0.2) # Val/Train split 80/20

print("")
print("Training paths:", train)
print("")
print("Testing paths:", test)
print("")
print("Validation paths:", val)


### change name of current dataset folder to dataset_date_time
now = datetime.now()
dt_string = now.strftime("_%d-%m-%Y %H-%M") # dd/mm/YY H:M:S
os.rename(sav_path,sav_path + dt_string)

#create new dataset folder from dataset_template
shutil.copytree(sav_path + "_template",sav_path)

### Script 
print("")
print("Starting script!")

j = 0
u = len(point_names)
# n = nrrd image path
for n in nrrd_image_paths:   
    # poti do točk za posamezno sliko
    p_paths = []
    for i in range(u):
        i = i + (j*u)
        p_paths.append(point_json_paths[i])
    j += 1

    # original slika + točke
    data_arr, orig_image_shape, orig_img_ratio = yolov8_functions.preprocess_image(n, filter_val)
    points = yolov8_functions.create_point_array(p_paths, faktor_preslikave)    # array original točk:

    # filename creation
    name = yolov8_functions.filename_creation(path, n, ".nrrd")
    filename = sav_path + "/PNGs/" + name

    # označi točke na sliki in shrani sliko v PNG
    yolov8_functions.save_image(orig_image_shape, square, points, data_arr, filename)

    # Loči podatke na train/test
    if(n in train):
        data = "train"
    elif(n in val):
        data = "val"
    else:
        data = "test"

    yolov8_functions.main_func(sav_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data)

print("Script is done!")