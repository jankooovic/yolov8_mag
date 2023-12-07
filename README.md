# Instructions:
0. Go to directory yolov8 and run yolov8_framework.py
___
1. Use script yolov8_preprocess.py to create a dataset for training
2. Use script yolov8_train.py to retrain the YOLOV8 model
3. Use script yolov8_predict.py to predict the points and evaluate model accuracy
4. Use script yolov8_evaluate.py to evaluate model accuracy

# Installing dependencies
pip install -r requirements.txt

# Paths:
- Original dataset -> ./data/RTG_dataset - this dataset is made by hand
- Dataset creation -> ./data/dataset
- Model training results -> /opt/homebrew/runs/pose/
- Prediction&Validation results -> ./data/predicted

# Config Files:
Path: ./config
- config_ALL.yaml

# Github repository
https://github.com/jankooovic/yolov8_mag.git