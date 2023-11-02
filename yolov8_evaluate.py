
""" Compare results from original RTG anotations and predicted RTG anotations """
import yolov8_functions
import json

# Dataset path:
json_test_path = "./data/dataset/JSON/"
json_predict_path = "./data/predicted/"
landmark_names = ['FHC', 'TKC', 'TML', 'FNOC', 'aF1']


# load Json files

# json paths to predicted folder
json_paths = yolov8_functions.get_dirs(json_predict_path)
json_paths_predicted = []
[json_paths_predicted.append(directory) for directory in json_paths if ".json" in str(directory)]

# remove path before
img_names_predicted =  []
for path in json_paths_predicted:
    image_name = yolov8_functions.filename_creation(path, "", sign="\\")    # change according to linux or windows
    img_names_predicted.append(image_name)

# json paths to test folder
json_paths = yolov8_functions.get_dirs(json_test_path)
json_paths_test = []
for path in json_paths:
    skipLoop = False

    for name in landmark_names:
        if name in path:
            skipLoop = True

    if skipLoop:
        continue

    json_paths_test.append(path)


# get only paths that are to be evaluated from test
json_paths_test_compare = []
for path in json_paths_test:
    skipLoop = True

    for name in img_names_predicted:
        if name in path:
            skipLoop = False

    if skipLoop:
        continue

    json_paths_test_compare.append(path)

# remove path before
img_names_test =  []
img_test_paths = []
for path in json_paths_test_compare:
    image_name = yolov8_functions.filename_creation(path, "", sign="\\")    # change according to linux or windows
    img_names_test.append(image_name)
    img_test_paths.append(path)

# get only evaluation paths from test paths
to_evaluate_test_paths = []
for idx, img_path in enumerate(img_test_paths):
    if img_names_test[idx] in img_names_predicted:
        to_evaluate_test_paths.append(img_path)
    else:
        continue

print("Test length:", len(img_names_test), "Test paths length:", len(img_test_paths), "Predicted Length:", len(img_names_predicted), "Test evaluation paths:", len(to_evaluate_test_paths))
for idx, path in enumerate(to_evaluate_test_paths):
    print("Test path:",path, "Predicted path:", json_paths_predicted[idx])

    # get point cooridnates from JSON 

    # Test points
    test_coordinates = 0
    with open(to_evaluate_test_paths[idx]) as f:
        data = json.load(f)
        test_coordinates = data['Point coordinates']

    # Predict points
    predicted_coordinates = 0
    with open(json_paths_predicted[idx]) as f:
        data = json.load(f)
        predicted_coordinates = data['Point coordinates']

    # compare point cooridnates - uporabim samo tiste, ki so si podobne ???
    

    # caculate error in pixels, percentage and x,y values comapred to image size

    # save to Json format for report

    # create report

