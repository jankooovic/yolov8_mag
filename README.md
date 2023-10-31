# Instructions:

0. Go to directory yolov8 - scripts are run fromyolov8 directory
1. Use script yolov8_preprocess.py to create a dataset for training
2. Use script yolov8_train.py to retrain the YOLOV8 model
3. Use script yolov8_predict_evaluate.py to predict the points and evaluate model accuracy

# Installing
pip install -r requirements.txt


# Paths:
- Original dataset -> ./data/RTG_dataset - this dataset is made by hand
- Dataset creation -> ./data/dataset
- Model training results -> /opt/homebrew/runs/pose/
- Prediction&Validation results -> ./data/predicted

# Config Files:
Path: Downloads/yolov8/data/dataset
- config_aF1.yaml
- config_ALL.yaml
- config_TKC.yaml
- config_TML.yaml
- config_FHC.yaml

# Github repository
https://github.com/jankooovic/yolov8_mag.git