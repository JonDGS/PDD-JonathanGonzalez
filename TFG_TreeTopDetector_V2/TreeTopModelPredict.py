from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train/weights/best.onnx')  # load an official detection model

results = model.predict(
    source="Tree CountingBigDataSet/test/images/cr_261969_4_174260_4_19_jpg.rf.9ab089103682630f20f937472785d651.jpg", 
    save = True,
    save_crop = True, 
    iou = 0.5,
    augment = True,
    show_labels = True,
    show_conf = True, 
    save_txt = True)  # predict on an image
