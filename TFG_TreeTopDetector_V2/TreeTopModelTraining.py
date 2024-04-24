from ultralytics import YOLO
import os

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("yolov8n.yaml")


model.train(
    data = os.path.join(os.getcwd(), "Tree CountingBigDataSet/data.yaml"), 
    epochs = 50, 
    cache=True, 
    plots = True, 
    overlap_mask = False, 
    bgr = 0.5
    )
# model.train(data = 'coco128.yaml', epochs = 5)
metrics = model.val()  # evaluate model performance on the validation set
# results = model("./valid/images/sg_262074_4_174314_4_19_jpg.rf.11ec272790fe2e805ef93ee622615f76.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
# path = model.export(format="tflite", keras=True)  # export the model to ONNX format