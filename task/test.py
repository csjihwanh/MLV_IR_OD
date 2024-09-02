from ultralytics import YOLOv10

flir_test_path= '/scratch/e1640a06/MLV_IR_OD/dataset/flir_adas/images_rgb_train/data/video-2BARff2EP7ZWkiF7n-frame-000432-pEpYGJT3PDodWjHHN.jpg'

def test():
    model = YOLOv10('checkpoints/yolov10x.pt')
    result = model.predict(flir_test_path)
    print('result is :',result[0].boxes)

if __name__ == '__main__':
    test()
