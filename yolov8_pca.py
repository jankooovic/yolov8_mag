""" Implementacija PCA metode za odstranitev napačnih koordinat """
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://visualstudiomagazine.com/articles/2021/10/20/anomaly-detection-pca.aspx

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import yolov8_functions
import json

# Dataset path:
predicted_path = "./data/predicted/"
test_path = "./data/dataset/JSON/"
test_images_path =  "./data/dataset/ALL/images/test/"
postprocess_path = "./data/postprocess/"
skipped_path = 'data/postprocess/skipped.json'
save_path = "./data/postprocess"
images_path = "./data/dataset/ALL/images/test/"
landmark_names = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA1', 'sFMDA2', 'sTMA1', 'sTMA2','TML']
point_names_all = ['FHC', 'aF1', 'FNOC', 'TKC', 'sFMDA', 'sTMA', 'TML']
skipped_path = "data/postprocess/skipped.json"
false_prediction = []
image_name = None

# create dataset archive
#yolov8_functions.dataset_archive(save_path)

# Load json files
#json_paths_predicted = [directory for directory in yolov8_functions.get_dirs(json_predict_path) if ".json" in str(directory)]
json_paths_predicted = [directory for directory in yolov8_functions.get_dirs(postprocess_path) if ".json" in str(directory)]
if skipped_path in json_paths_predicted:
    json_paths_predicted.remove(skipped_path)

# get only paths that are to be evaluated from test
json_paths_test = [path for path in yolov8_functions.get_dirs(test_path) if not any(name in path for name in point_names_all)]

# get test images
test_images = yolov8_functions.get_dirs(test_images_path)
img_names_test =  [yolov8_functions.filename_creation(path, "") for path in test_images]
for i, img in enumerate(img_names_test):
    img_names_test[i] = img.replace(".jpg","")

# get test json files names
json_names_test =  [yolov8_functions.filename_creation(path, "") for path in json_paths_test]
for i, img in enumerate(json_names_test):
    json_names_test[i] = img.replace(".json","")

# get only evaluation paths from test paths
to_evaluate_json_paths = []
for i, p in enumerate(json_names_test):
    for j, im in enumerate(img_names_test):
        if p == im:
            to_evaluate_json_paths.append(json_paths_test[i])

# remove images with false predictions
to_skip = []
with open(skipped_path) as f:
        data = json.load(f)
        to_skip = (data['False predictions'])

to_skip =  [img_name for img_name in to_skip]
for i, name in enumerate(to_skip):
    to_skip_name = name.replace(".jpg","")
    to_skip[i] = "data/dataset/JSON/" + to_skip_name + ".json"

for skip in to_skip:
    print(skip)
    if skip in to_evaluate_json_paths:
        to_evaluate_json_paths.remove(skip)

# sort paths:
json_paths_predicted = sorted(json_paths_predicted)
json_paths_test = sorted(json_paths_test)
to_evaluate_json_paths = sorted(to_evaluate_json_paths)

#print("Length test:", len(to_evaluate_json_paths), "Predicted test:", len(json_paths_predicted))

predicted_coordinates = []
test_coordinates = []

for idx, path in enumerate(to_evaluate_json_paths):
    skip = False
    print("Path:", path)

    # Test points json
    point_names = []
    img_size = []
    with open(path) as f:
        data = json.load(f)
        for coord in point_names_all:
            if coord == 'sTMA':
                stma1_x = data[coord][0]
                stma1_y = data[coord][1]
                stma2_x = data[coord][2]
                stma2_y = data[coord][3]
                stma1 = [stma1_x, stma1_y]
                stma2 = [stma2_x, stma2_y]
                test_coordinates.append(stma1)
                point_names.append('sTMA1')
                test_coordinates.append(stma2)
                point_names.append('sTMA2')
            elif coord == 'sFMDA':
                stma1_x = data[coord][0]
                stma1_y = data[coord][1]
                stma2_x = data[coord][2]
                stma2_y = data[coord][3]
                stma1 = [stma1_x, stma1_y]
                stma2 = [stma2_x, stma2_y]
                test_coordinates.append(stma1)
                point_names.append('sFMDA1')
                test_coordinates.append(stma2)
                point_names.append('sFMDA2')
            else:
                test_coordinates.append(data[coord])
                point_names.append(coord)
        img_size =  data['Image_size']  # x,y are swapped
        img_size = [img_size[1], img_size[0]]

    # Predicted points json
    path = json_paths_predicted[idx]
    with open(path) as f:
        data = json.load(f)
        for name in landmark_names:
            predicted_coordinates.append(data[name])


print("Predicted coordinates", predicted_coordinates)
print("Test coordinates", predicted_coordinates)

from sklearn.decomposition import PCA

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# The fit learns some quantities from the data, most importantly the "components" and "explained variance":
pca = PCA().fit(test_coordinates)

# PCA components
print(pca.components_)
# PCA variance
print(pca.explained_variance_)

# noisy data
pca = PCA(0.50).fit(predicted_coordinates)
print("PCA predicted components at 50%:",pca.n_components_)

components = pca.transform(predicted_coordinates)
filtered = pca.inverse_transform(components)
print(filtered)

# prikaz točk na slikah per point - evaluate fora
# izračun razlik med točkami in če jih dejansko doda/odstrani še preveri
# PCA uporabim na način, da ga nučim za posamezno točko, kje se mora nahajti ali na vseh skupaj??? -> premisli
# preberi še enkrat tekst, da boš razumel, kaj se gre