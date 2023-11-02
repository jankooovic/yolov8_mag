""" Use to execute all steps in the framework. """
import subprocess

print("Starting framework procedure ...")

# Run yolov8_preprocess.py
with open("yolov8_preprocess.py") as f:
    exec(f.read())
print("Executing yolov8_preprocess.py ...")
subprocess.run(["python", "yolov8_preprocess.py"])

# Run yolov8_train.py
with open("yolov8_train.py") as f:
    exec(f.read())
print("Executing yolov8_train.py ...")
subprocess.run(["python", "yolov8_train.py"])

# Run yolov8_predict.py
with open("yolov8_predict.py") as f:
    exec(f.read())
print("Executing yolov8_predict.py ...")
subprocess.run(["python", "yolov8_predict.py"])

# Run yolov8_evaluate.py
with open("yolov8_evaluate.py") as f:
    exec(f.read())
print("Executing yolov8_evaluate.py ...")
subprocess.run(["python", "yolov8_evaluate.py"])

print("Finished framework procedure!")