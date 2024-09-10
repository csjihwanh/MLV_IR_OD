from test import convert_to_submit

import argparse
import os
import json
import cv2

from ultralytics import YOLOv10
    
def test(args):

    model = YOLOv10(args.checkpoint)

    result_dict_list = []

    result = model.predict(args.img_path)
    convert_to_submit(result_dict_list, result[0].boxes, 'test')
    
    image = cv2.imread(args.img_path)
    if image is None:
        print(f"Error loading image: {args.img_path}")
        return

    # Classes names
    class_names = {
        0: "person",
        1: "car",
        2: "truck",
        3: "bus",
        4: "bicycle",
        5: "bike",
        6: "extra_vehicle",
        7: "dog"
    }


    for result_dict in result_dict_list:
        category_id = result_dict["category_id"]
        bbox = result_dict["bbox"] # x_left, y_left, w, h 
        score = result_dict["score"]

    # Convert bbox from center x, center y, width, height to x_left, y_top, x_right, y_bottom
        x_left, y_top, width, height = bbox
        x_left = int(x_left)
        y_top = int(y_top)
        x_right = int(x_left + width)
        y_bottom = int(y_top + height)

        # Draw the rectangle (bounding box)
        cv2.rectangle(image, (x_left, y_top), (x_right, y_bottom), color=(0, 255, 0), thickness=2)

        # Put label with class name and confidence score
        label = f"{class_names[category_id]}: {score:.2f}"
        cv2.putText(image, label, (x_left, y_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with bounding boxes to the local directory
    output_image_path = 'output_with_bboxes.png'
    cv2.imwrite(output_image_path, image)
    print(f"Image with bounding boxes saved to {output_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, help="img_path", required=True)
    parser.add_argument('--checkpoint', type=str, help="checkpoint", required=False, default='checkpoints/yolov10x.pt' )
    parser.add_argument('--save_dir', type=str, help="save_dir", required=False, default='result.json')

    args = parser.parse_args()
    print(args)

    test(args)