import argparse

from ultralytics import YOLOv10

def train(args):

    model = YOLOv10('checkpoints/yolov10x.pt')

    if args.dataset == 'hscai':
        dataset = 'hscai.yaml'
    else :
        raise ValueError('dataset {dataset} is invalid')
    
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
