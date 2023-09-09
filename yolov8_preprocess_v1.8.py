### Imports: 
import yolov8_functions
from sklearn.model_selection import train_test_split

### Variables: ###

# Dataset path:
path = "/Users/jankovic/Downloads/mag_data/Data/RTG_dataset/"
sav_path = './data/dataset'

# point names array
point_names = ['FHC', 'TKC', 'TML', 'aF1']

# obdelava variables:
filter_val = 10000
faktor_preslikave = 3.6

# square size - matplotlib + veliksot za kvadrat učenja
square = 0.1

# number of parts on image
num_parts = 3

### Directories and images: ###

## poti do direktorijev, kjer so nrrd datoteke
dirs = yolov8_functions.get_dirs(path)

## poti do nrrd slik
nrrd_image_paths = yolov8_functions.get_nrrd_paths(dirs)

## poti do json datotek s podatki o točkah
point_json_paths = yolov8_functions.get_point_json_paths(dirs, point_names)

### Make train, test and validate groups ###
"""
Split:
Test = 20%
Train = 80%
Validate = 20% of Train
"""

# Size of entire Dataset
all_num = len(nrrd_image_paths)

# Train/Test split 80/20
train,test=train_test_split(nrrd_image_paths,test_size=0.2)

# Val/Train split 80/20
train,val=train_test_split(train,test_size=0.2)

print("______")
print("Training paths:", train)
print("")
print("Testing paths:", test)
print("")
print("Validation paths:", val)

##### Make dataset ####

j = 0
u = len(point_names)
# n = nrrd image path
for n in nrrd_image_paths: 

    print("______")
    print("Nrrd image path:", n)
    
    # poti do točk za posamezno sliko
    p_paths = []
    for i in range(u):
        i = i + (j*u)
        p_paths.append(point_json_paths[i])
    j += 1

    # original slika
    data_arr, orig_image_shape, orig_img_ratio = yolov8_functions.preprocess_image(n, filter_val)

    # array original točk:
    points = yolov8_functions.create_point_array(p_paths, faktor_preslikave)
    #print("Point coordinates", points)

    # filename creation
    name = yolov8_functions.filename_creation(path, n, ".nrrd")
    filename = sav_path + "/PNGs/" + name

    # označi točke na sliki in shrani sliko v PNG
    yolov8_functions.show_save_full_image(orig_image_shape, square, points, data_arr, filename)

    # Loči podatke na train/test
    if(n in train):
        print("Training dataset")
        data = "train"
        yolov8_functions.main_func(sav_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data)
    elif(n in val):
        print("Validation dataset")
        data = "val"
        yolov8_functions.main_func(sav_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data)
    else:
        print("Testing dataset")
        data = "test"
        yolov8_functions.main_func(sav_path, name, data_arr, point_names, points, orig_image_shape, square, orig_img_ratio, data)

print("")
print("#################")
print("Script is done!")
print("")