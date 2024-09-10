import argparse
import os
import json
import re

from ultralytics import YOLOv10

def get_img_list(path):
    
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_list.sort(key=lambda f: [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', f)])
    return file_list 

# submit format: {"image_id": "val_0", "category_id": 5, "bbox": [359.361, 207.394, 22.972, 43.519], "score": 0.01585}, 
def convert_to_submit(result_dict_list, bbox_obj, image_id):

    for idx in range(len(bbox_obj.cls)):
        result_dict = {}
        result_dict["image_id"] = image_id
        result_dict["category_id"] = int(bbox_obj.cls[idx].detach().item())
        result_dict["bbox"] = bbox_obj.xyxy[idx].detach().tolist()[:2] + bbox_obj.xywh[idx].detach().tolist()[2:]
        result_dict["score"] = bbox_obj.conf[idx].detach().item()
        result_dict_list.append(result_dict)
    
def test(args):

    model = YOLOv10(args.checkpoint)

    if args.dataset == 'hscai':
        dataset_path = 'datasets/hscai/images/test_open'
    else :
        raise ValueError('dataset {dataset} is invalid')

    img_list = get_img_list(dataset_path)

    result_dict_list = []

    for img_id in img_list:
        img_path = os.path.join(dataset_path, img_id)
        result = model.predict(img_path, conf=0.01, iou=0.6)
        convert_to_submit(result_dict_list, result[0].boxes, img_id[:-4])

    with open(args.save_dir, 'w') as json_file:
        json.dump(result_dict_list, json_file, indent=4) 
    print(f'result saved at {args.save_dir}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help="dataset", required=True)
    parser.add_argument('--checkpoint', type=str, help="checkpoint", required=False, default='checkpoints/yolov10x.pt' )
    parser.add_argument('--save_dir', type=str, help="save_dir", required=False, default='result.json')

    args = parser.parse_args()
    print(args)

    test(args)
