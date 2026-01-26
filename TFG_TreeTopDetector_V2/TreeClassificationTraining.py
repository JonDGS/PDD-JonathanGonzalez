from ultralytics import YOLO
import os

# model = YOLO("yolov8m.yaml")
model = YOLO("yolo11n-cls.pt")

model.train(
    data = os.path.join(os.getcwd(), "PDD-JonathanGonzalez/Tree Counting BigDS"),
    epochs = 50,
    imgsz=640,
    batch = -1, 
    plots = True, 
    overlap_mask = False,
    workers=2,
    #augmentation parameters
    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
    degrees = 0,
    translate = 0.1,
    scale = 0.5,
    shear= 0,
    perspective =0,
    flipud = 0,
    fliplr = 0.5,
    bgr = 0,
    mosaic = 0,
    mixup = 0.1,
    copy_paste = 0,
    erasing = 0.4,
    crop_fraction = 1
    )

metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format