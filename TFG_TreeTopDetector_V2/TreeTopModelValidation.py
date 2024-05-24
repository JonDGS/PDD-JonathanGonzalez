from ultralytics import YOLO
import os

# Load a model
model = YOLO('runs/detect/E-9/weights/best.onnx')  # load an official detection model

results = model.val(
    data = os.path.join(os.getcwd(), "Tree Counting Original/data.yaml"),
    save = True,
    save_json = True, 
    save_hybrid = True, 
    iou = 0.5,
    plots = True)  # predict on an image
