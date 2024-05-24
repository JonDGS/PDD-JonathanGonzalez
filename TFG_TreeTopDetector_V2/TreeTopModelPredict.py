from ultralytics import YOLO
import os


# Load a model
model = YOLO('runs/detect/E-9/weights/best.onnx')  # load an official detection model


results = model.predict(
    source="Tree Counting Original/test/images/test_10.jpg", 
    show_conf = True,
    visualize = True, 
    save = True,
    save_crop = False, 
    iou = 0.5,
    conf = 0.25,
    augment = True,
    show_labels = True,
    save_txt = True)  # predict on an image
