from roboflow import Roboflow
rf = Roboflow(api_key="qiw1aXXrTf6ehRgcfqdn")
project = rf.workspace("project-s402o").project("tree-counting-qiw3h")
version = project.version(1)
dataset = version.download("tfrecord")

model = project.version(1).model

response = model.predict(image_path="Tree CountingBigDataSet/valid/images/cd_261945_4_174271_4_19_jpg.rf.52edba2d0d2ba0fddab731467add975a.jpg", confidence=40, overlap=30,).json()

for pred in response['predictions']:
    print(pred['class'])