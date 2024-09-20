import json
from tqdm import tqdm

# Function to compute IoU between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the (x, y)-coordinates of the intersection box
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Compute the area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Compute the IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def filter_predictions(predictions, iou_threshold=0.5, score_threshold=0.5, inclusion_threshold=0.95):
    filtered_predictions = []

    # Group predictions by image (category is ignored)
    pred_by_image = {}
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in pred_by_image:
            pred_by_image[image_id] = []
        pred_by_image[image_id].append(pred)

    # Process each image
    for image_id, pred_list in pred_by_image.items():
        # Check all pairs of bounding boxes in the same image
        keep_predictions = pred_list[:]
        for i, pred1 in enumerate(pred_list):
            for j, pred2 in enumerate(pred_list):
                if i != j:
                    iou = compute_iou(pred1["bbox"], pred2["bbox"])
                    
                    # IoU check for duplicate removal
                    if iou > iou_threshold:
                        # Remove the one with the lower score if score < threshold
                        if pred1["score"] < score_threshold:
                            if pred1 in keep_predictions:
                                keep_predictions.remove(pred1)
                        elif pred2["score"] < score_threshold:
                            if pred2 in keep_predictions:
                                keep_predictions.remove(pred2)
                    
                    # Check if one box is almost entirely inside another (95% or more)
                    area1 = compute_area(pred1["bbox"])
                    area2 = compute_area(pred2["bbox"])
                    
                    if iou > inclusion_threshold:
                        # If pred1 is almost entirely inside pred2 and score is lower, remove pred1
                        if area1 / area2 > inclusion_threshold and pred1["score"] < pred2["score"]:
                            if pred1 in keep_predictions:
                                keep_predictions.remove(pred1)
                        # If pred2 is almost entirely inside pred1 and score is lower, remove pred2
                        elif area2 / area1 > inclusion_threshold and pred2["score"] < pred1["score"]:
                            if pred2 in keep_predictions:
                                keep_predictions.remove(pred2)

        # Ensure at least one prediction is left
        if not keep_predictions:
            keep_predictions = [max(pred_list, key=lambda p: p["score"])]

        # Add the filtered predictions for this image
        filtered_predictions.extend(keep_predictions)

    return filtered_predictions

# Function to compute the area of a bounding box
def compute_area(bbox):
    _, _, w, h = bbox
    return w * h

# Function to check if bbox1 is contained in bbox2 (returns the percentage of bbox1 inside bbox2)
def compute_containment(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Compute the area of bbox1 (contained bbox)
    bbox1_area = compute_area(bbox1)
    
    # Compute percentage of bbox1 contained inside bbox2
    if bbox1_area > 0:
        containment = inter_area / bbox1_area
    else:
        containment = 0

    return containment


# Function to filter out predictions based on containment and score threshold within the same image
def filter_contained_predictions(predictions, containment_threshold=0.95, score_threshold=0.5):
    filtered_predictions = []

    # Group predictions by image_id
    pred_by_image = {}
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in pred_by_image:
            pred_by_image[image_id] = []
        pred_by_image[image_id].append(pred)

    # Iterate over each image's predictions
    for image_id, preds in tqdm(pred_by_image.items(), desc="Processing Images"):
        # Compare bounding boxes within the same image
        keep_predictions = preds[:]
        for i, pred1 in enumerate(preds):
            for j, pred2 in enumerate(preds):
                if i != j:
                    containment = compute_containment(pred1["bbox"], pred2["bbox"])

                    # If pred1 is 95% contained inside pred2 and its score is lower than threshold
                    if containment > containment_threshold and pred1["score"] < score_threshold:
                        if pred1 in keep_predictions:
                            keep_predictions.remove(pred1)
        
        # Add the remaining predictions for this image to the filtered list
        filtered_predictions.extend(keep_predictions)
    
    return filtered_predictions

# Function to calculate the vertical center of a bounding box
def get_bbox_vertical_center(bbox):
    _, y, _, h = bbox  # Only y and height (h) are relevant
    center_y = y + h / 2
    return center_y

# Function to calculate the absolute distance between two vertical positions (image center and bbox center)
def calculate_vertical_distance(y1, y2):
    return abs(y1 - y2)

# Function to filter bboxes based on their vertical distance from the image center
def filter_by_vertical_distance(predictions, image_height, distance_threshold, score_threshold):
    filtered_predictions = []

    # Image center (vertical) is always at height / 2
    image_center_y = image_height / 2
    
    # Iterate over all the predictions
    for pred in tqdm(predictions, desc="Processing BBoxes by Vertical Distance"):
        bbox_center_y = get_bbox_vertical_center(pred["bbox"])

        # Calculate the distance between the bbox center and the vertical image center
        vertical_distance = calculate_vertical_distance(bbox_center_y, image_center_y)

        # If the vertical distance is greater than the threshold and the score is lower than the threshold, remove it
        if vertical_distance > distance_threshold and pred["score"] < score_threshold:
            continue  # Don't keep this bbox
        else:
            filtered_predictions.append(pred)
    
    return filtered_predictions

def filter_by_category_and_score(predictions):
    filtered_predictions = []

    for pred in tqdm(predictions, desc="Filtering by Category and Score"):
        # Check if category_id is 6 and score is less than 0.05
        if pred["category_id"] == 6 and pred["score"] < 0.05:
            continue  # Discard this prediction
        else:
            filtered_predictions.append(pred)
    
    return filtered_predictions

# Load the predictions from a JSON file
def load_predictions(file_path):
    with open(file_path, 'r') as f:
        predictions = json.load(f)
    return predictions

# Save the filtered predictions to a JSON file
def save_filtered_predictions(filtered_predictions, output_path):
    with open(output_path, 'w') as f:
        json.dump(filtered_predictions, f, indent=4)

# Main code
input_file = '/workspace/MLV_IR_OD/results/result_uni_last_epoch256_1e2.txt'  # Replace with your actual file path
output_file = '/workspace/MLV_IR_OD/results/result_uni_last_epoch256_fin1.txt'
iou_threshold = 0.95  # Set your IoU threshold
score_threshold = 0.1  # Set your score threshold

containment_threshold = 0.95
containment_score_threshold = 0.2

image_height=480
distance_threshold = 100  # Set the vertical distance threshold
distance_score_threshold = 0.05  # Set the score threshold

# Load predictions
predictions = load_predictions(input_file)

# Filter predictions based on IoU and score thresholds
filtered_predictions = filter_predictions(predictions, iou_threshold=iou_threshold, score_threshold=score_threshold)
#filtered_predictions = filter_contained_predictions(predictions, containment_threshold=containment_threshold, score_threshold=containment_score_threshold)
filtered_predictions = filter_by_vertical_distance(filtered_predictions, image_height, distance_threshold, distance_score_threshold)
filtered_predictions = filter_by_category_and_score(filtered_predictions)

# Save the filtered predictions
save_filtered_predictions(filtered_predictions, output_file)

print(f"Filtered predictions saved to {output_file}")
