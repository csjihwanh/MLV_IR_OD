from ultralytics import YOLOv10

model = YOLOv10(model='yolov10mlv.yaml')
model.load('/workspace/MLV_IR_OD/runs/detect/train26/weights/best.pt')

print(model)