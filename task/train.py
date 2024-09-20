import argparse
import torch

from ultralytics import YOLOv10

def train(args):

    if args.model == 'yolov10x':

        if args.ckpt == 'None':
            model = YOLOv10('checkpoints/yolov10x.pt')
            pretrained= False
        else:
            model = YOLOv10(args.ckpt)
            pretrained=True

    if args.model == 'yolov10mlv':
        model = YOLOv10('runs/detect/train40/weights/last.pt')
        checkpoint = torch.load('/workspace/MLV_IR_OD/runs/detect/train26/weights/last.pt')
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=False)
        pretrained=True

        if args.freeze:
            for name, param in model.named_parameters():
                # Freeze all layers except AIFI modules
                if 'AIFI' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True 
            
            for name, param in model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

    dual_train = False


    if args.dataset == 'hscai':
        dataset = 'hscai.yaml'
    elif args.dataset == 'flir_adas':
        dataset = 'flir_adas.yaml'
    elif args.dataset == 'vehicles-and-pedestrians':
        dataset = 'vehicles-and-pedestrians.yaml'
    elif args.dataset == 'flir2hscai':
        dual_train = True
    elif args.dataset == 'unified_dataset':
        dataset = 'unified_dataset.yaml'
    
    else :
        raise ValueError('dataset {dataset} is invalid')
    
    if dual_train:
        print('dual train mode is activated')
        model.train(data='flir_adas.yaml', pretrained = pretrained, optimizer=args.optimizer, lr0= args.lr0, lrf=args.lrf, cos_lr=args.cos_lr, resume= args.resume, device =args.device, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
        model.train(data='hscai.yaml', optimizer=args.optimizer, lr0= args.lr0, lrf=args.lrf, cos_lr=args.cos_lr, resume= args.resume, device =args.device, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)

    else: 
        model.train(data=dataset,pretrained = pretrained, optimizer=args.optimizer, lr0= args.lr0, lrf=args.lrf, cos_lr=args.cos_lr, resume= args.resume, device =args.device, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, help="epoch", required=True)
    parser.add_argument('--device', type=str, help="device", default='0', required=False)
    parser.add_argument('--lr0', type=float, help="", default=1e-3, required=False)
    parser.add_argument('--lrf', type=float, help="", default=1e-4, required=False)
    parser.add_argument('--cos_lr', action='store_true', help="", default=False, required =False)
    parser.add_argument('--resume', action='store_true', help="resume training from last checkpoint")
    parser.add_argument('--batch', type=int, help="batch", required=True)
    parser.add_argument('--imgsz', type=int, help="imgsz", default=640, required=False)
    parser.add_argument('--dataset', type=str, help="dataset", required=True)
    parser.add_argument('--optimizer', type=str, help="optimizer", default = 'auto', required=False)
    parser.add_argument('--ckpt', type=str, help="", default='checkpoints/yolov10x.pt', required=False)
    parser.add_argument('--model', type=str, help="", default='yolov10x', required=False)
    parser.add_argument('--freeze', action='store_true', help="resume training from last checkpoint")

    args = parser.parse_args()
    print(args)

    train(args)
