from ultralytics import YOLO
import os


# model = YOLO("yolov8m.yaml")
model = YOLO("yolov8n.yaml")


model.train(
    data = os.path.join(os.getcwd(), "Tree Counting Original/data.yaml"),
    epochs = 5,
    batch = -1, 
    plots = True, 
    overlap_mask = True,
    workers=8,
    #augmentation parameters
    hsv_h = 0.2,
    hsv_s = 0.2,
    hsv_v = 0.2,
    degrees = 0,
    translate = 0.2,
    scale = 0.5,
    shear= 0,
    perspective =0,
    flipud = 0.2,
    fliplr = 0.2,
    bgr = 0.5,
    mosaic = 0.5,
    mixup = 0.5,
    copy_paste = 0.2,
    erasing = 0.4,
    crop_fraction = 1
    )

metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format
