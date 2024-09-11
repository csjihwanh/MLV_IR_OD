import argparse

from ultralytics import YOLOv10

def train(args):

    model = YOLOv10('checkpoints/yolov10x.pt')
    dual_train = False

    if args.dataset == 'hscai':
        dataset = 'hscai.yaml'
    elif args.dataset == 'flir_adas':
        dataset = 'flir_adas.yaml'
    elif args.dataset == 'flir2hscai':
        dual_train = True
    else :
        raise ValueError('dataset {dataset} is invalid')
    
    if dual_train:
        print('dual train mode is activated')
        model.train(data='flir_adas.yaml', device =args.device, epochs=100, batch=args.batch, imgsz=args.imgsz)
        model.train(data='hscai.yaml', device =args.device, epochs=100, batch=args.batch, imgsz=args.imgsz)

    else: 
        model.train(data=dataset, device =args.device, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, help="epoch", required=True)
    parser.add_argument('--device', type=str, help="device", default='0', required=False)
    parser.add_argument('--batch', type=int, help="batch", required=True)
    parser.add_argument('--imgsz', type=int, help="imgsz", default=640, required=False)
    parser.add_argument('--dataset', type=str, help="dataset", required=True)

    args = parser.parse_args()

    train(args)
