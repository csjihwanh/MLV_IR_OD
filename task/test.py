import argparse
import os
import json
import re
from collections import defaultdict
import math

from ultralytics import YOLOv10

def get_img_list(path):
    
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_list.sort(key=lambda f: [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', f)])
    return file_list 

# submit format: {"image_id": "val_0", "category_id": 5, "bbox": [359.361, 207.394, 22.972, 43.519], "score": 0.01585}, 
def convert_to_submit(result_dict_list, bbox_obj, image_id, score_by_category):

    for idx in range(len(bbox_obj.cls)):
        result_dict = {}
        result_dict["image_id"] = image_id
        result_dict["category_id"] = int(bbox_obj.cls[idx].detach().item())
        result_dict["bbox"] = bbox_obj.xyxy[idx].detach().tolist()[:2] + bbox_obj.xywh[idx].detach().tolist()[2:]
        result_dict["score"] = bbox_obj.conf[idx].detach().item()
        result_dict_list.append(result_dict)

        # Update score by category
        category_id = result_dict["category_id"]
        score_by_category[category_id]['scores'].append(result_dict["score"])
    
    
def test(args):

    model = YOLOv10(args.checkpoint)

    if args.dataset == 'hscai':
        dataset_path = 'datasets/hscai/images/test_open'
    else :
        raise ValueError('dataset {dataset} is invalid')

    img_list = get_img_list(dataset_path)

    result_dict_list = []

    score_by_category = defaultdict(lambda: {'scores': []})
    
    for img_id in img_list:
        img_path = os.path.join(dataset_path, img_id)
        result = model.predict(img_path, conf=args.conf_threshold, device=args.device)
        convert_to_submit(result_dict_list, result[0].boxes, img_id[:-4], score_by_category)


    print("\nAverage confidence score and std by category_id:")
    thresholds = {}

    for category_id, values in score_by_category.items():
        if len(values['scores']) > 0:
            avg_score = sum(values['scores']) / len(values['scores'])
            # Calculate standard deviation
            variance = sum((x - avg_score) ** 2 for x in values['scores']) / len(values['scores'])
            std_dev = math.sqrt(variance)
            print(f"Category {category_id}: Average = {avg_score:.4f}, Std Dev = {std_dev:.4f}")

            # Custom threshold: avg_score - 0.5 * std_dev
            thresholds[category_id] = avg_score - 0.5 * std_dev
        else:
            print(f"Category {category_id}: No instances found.")

            # Custom threshold: avg_score - 0.5 * std_dev
            thresholds[category_id] = 0.01

    if args.custom_threshold :
        ### custom threshold discard
        filtered_result_dict_list = []
        for result in result_dict_list:
            category_id = result['category_id']
            score = result['score']
            # Discard if score is below the threshold for the category
            if score >= thresholds[category_id]:
                filtered_result_dict_list.append(result)

        print(f"\nNumber of results before filtering: {len(result_dict_list)}")
        print(f"Number of results after filtering: {len(filtered_result_dict_list)}")
        result_dict_list = filtered_result_dict_list

    with open(os.path.join('results',args.save_dir), 'w') as json_file:
        json.dump(result_dict_list, json_file, indent=4) 
    print(f'result saved at {args.save_dir}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help="dataset", required=True)
    parser.add_argument('--device', type=str, help="device", default='0', required=False)
    parser.add_argument('--checkpoint', type=str, help="checkpoint", required=False, default='checkpoints/yolov10x.pt' )
    parser.add_argument('--save_dir', type=str, help="save_dir", required=False, default='result.json')
    parser.add_argument('--custom_threshold', action='store_true', help="set confidence threshold according to the conf distribution", required=False, default=False)
    parser.add_argument('--conf_threshold', type=float, help="confidence threshold", required=False, default=0.01)

    args = parser.parse_args()
    print(args)

    test(args)
