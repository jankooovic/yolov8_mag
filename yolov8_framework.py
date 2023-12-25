""" Use to execute all steps in the framework. """
import subprocess

print("Starting framework procedure ...")

"""
# Run yolov8_preprocess.py
print("Executing yolov8_preprocess.py ...")
with open("yolov8_preprocess.py") as f:
    exec(f.read())
f.close()
#subprocess.run(["python", "yolov8_preprocess.py"])

# Run yolov8_train.py
print("Executing yolov8_train.py ...")
with open("yolov8_train.py") as f:
    exec(f.read())
f.close() 
"""

# Run yolov8_predict.py
print("Executing yolov8_predict.py ...")
with open("yolov8_predict_pca.py") as f:
    exec(f.read())
f.close()
"""
# Run yolov8_postprocess.py
print("Executing yolov8_postprocess.py ...")
with open("yolov8_postprocess.py") as f:
    exec(f.read())
f.close()
"""
# Run yolov8_evaluate.py
print("Executing yolov8_evaluate.py ...")
with open("yolov8_evaluate.py") as f:
    exec(f.read())
f.close()

print("Finished framework procedure!")