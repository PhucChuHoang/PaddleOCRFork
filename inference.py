from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sortedcontainers import SortedDict

det_model_dir = './inference/det/PP-OCRv5_server_det_infer'
rec_model_dir = './inference/customized/svtr_base'
rec_char_dict_path = './ppocr/utils/dict/casia_hwdb_dict.txt'
nom_dict_path = 'ppocr/utils/dict/nom_dict.txt'

ocr = PaddleOCR(
    det_model_dir=det_model_dir,
    rec_model_dir=rec_model_dir,
    rec_char_dict_path=rec_char_dict_path,
    rec_image_shape='3,48,48',
    rec_algorithm='SVTR',
    det_algorithm='DB',
    use_angle_cls=False,
    use_space_char=True,
    use_gpu=True,
    show_log=False,
    drop_score=0,
)

def char2code(ch):
    pos = ord(ch) - 0xF0000
    return pos

def visualize_results(image, result, output_path='visualized_output.jpg'):
    with open(nom_dict_path, 'r', encoding='utf-8') as f:
        nom_dict = f.read().splitlines()
    
    # Convert OpenCV image to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font that supports Vietnamese
    try:
        # Try to find a system font that supports Vietnamese
        font_size = 20
        font = ImageFont.truetype("arial.ttf", font_size)  # Windows default font
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Draw detection boxes and recognition results
    for idx, line in enumerate(result):
        # Extract box coordinates
        boxes = line[0]
        box = np.array(boxes).astype(np.int32).reshape(-1, 2)
        
        # Draw polygon around text area (convert points to tuple for PIL)
        points = [(point[0], point[1]) for point in box]
        draw.line(points + [points[0]], fill=(0, 255, 0), width=2)
        
        # Get text and confidence
        text, confidence = line[1]
        
        # Get the corresponding Vietnamese text
        viet_text = nom_dict[char2code(text) - 1]
        
        # Display text above the box
        text_position = (int(box[0][0]), int(box[0][1] - 30))
        label = f"{viet_text} ({confidence:.2f})"
        
        # Add white background for text readability
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle(
            [text_position[0], text_position[1], text_position[0] + text_width, text_position[1] + text_height],
            fill=(255, 255, 255)
        )
        
        # Draw the text with full Unicode support
        draw.text(text_position, label, fill=(255, 0, 0), font=font)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the visualization
    pil_image.save(output_path)
    print(f"Visualization saved to {output_path}")

def cluster_columns(boxes, eps_w_multiplier=0.6, is_vertical=True):
    """
    Group character boxes into columns using 1D clustering on x-centroids.

    Parameters:
      boxes: array-like of shape (N, 4) with [x_min, y_min, x_max, y_max]
      eps_w_multiplier: multiple of median char-width for DBSCAN eps

    Returns:
      labels: integer cluster label per box (0..N-1)
    """
    # Convert to numpy array
    boxes = np.asarray(boxes)
    if is_vertical:
        x_min, _, x_max, _ = boxes.T

        # Compute char widths
        widths  = x_max - x_min
        W = np.median(widths)

        # Compute X-centroids and Y-centroids
        x_centroids = (x_min + x_max) / 2

        # 1D clustering via DBSCAN
        eps = eps_w_multiplier * W
        clustering = DBSCAN(eps=eps, min_samples=3, n_jobs=-1)
        raw_labels = clustering.fit_predict(x_centroids.reshape(-1, 1))

        # Compute mean-X of each cluster (skip noise = -1 if any)
        unique = [lbl for lbl in np.unique(raw_labels) if lbl >= 0]
        mean_x = {lbl: x_centroids[raw_labels == lbl].mean() for lbl in unique}

        # Sort clusters by mean-X descending (right-most first)
        sorted_by_right = sorted(unique, key=lambda lbl: mean_x[lbl], reverse=True)
    else:
        _, y_min, _, y_max = boxes.T

        # Compute char heights
        heights = y_max - y_min
        H = np.median(heights)

        # Compute Y-centroids
        y_centroids = (y_min + y_max) / 2

        # 1D clustering via DBSCAN
        eps = eps_w_multiplier * H
        clustering = DBSCAN(eps=eps, min_samples=2, n_jobs=-1)
        raw_labels = clustering.fit_predict(y_centroids.reshape(-1, 1))

        # Compute mean-Y of each cluster (skip noise = -1 if any)
        unique = [lbl for lbl in np.unique(raw_labels) if lbl >= 0]
        mean_y = {lbl: y_centroids[raw_labels == lbl].mean() for lbl in unique}

        # Sort clusters by mean-Y ascending (top-most first)
        sorted_by_right = sorted(unique, key=lambda lbl: mean_y[lbl])

    # Build a remapping: old_label → new_label
    remap = {old: new for new, old in enumerate(sorted_by_right)}

    # Apply remapping (noise stays -1)
    new_labels = np.array([remap[lbl] if lbl >= 0 else -1 for lbl in raw_labels])

    return new_labels

# Process the image
img_path = 'test_images/page_102.png'
img = cv2.imread(img_path)
result = ocr.ocr(img, det=True, cls=False)

altered_result = []
for line in result[0]:
    coords = (line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1])
    altered_result.append([coords, line[1]])

is_vertical = True  # Set to True for horizontal text, False for vertical text
# Cluster the text regions into columns
labels = cluster_columns(np.array([line[0] for line in altered_result]), is_vertical=is_vertical)
# Combine results based on clustering
clustered_result = SortedDict()
for label, line in zip(labels, altered_result):
    if label not in clustered_result:
        clustered_result[label] = []
    clustered_result[label].append(line)

if is_vertical:
    # sort the clustered results by y position
    for label in clustered_result:
        clustered_result[label].sort(key=lambda x: x[0][1])  # Sort by y_min
else:
    # sort the clustered results by x position
    for label in clustered_result:
        clustered_result[label].sort(key=lambda x: x[0][0])

# Convert clustered results back to the format expected by visualize_results
print(f"Found {len(clustered_result)} columns")
print(f"Found {len(altered_result)} text regions")
with open(nom_dict_path, 'r', encoding='utf-8') as f:
    nom_dict = f.read().splitlines()
text, confidence = line[1][0], line[1][1]
viet_text = nom_dict[char2code(text) - 1]
for label, lines in clustered_result.items():
    if is_vertical:
        print(f"Column {label}: {len(lines)} text regions")
    else:
        print(f"Row {label}: {len(lines)} text regions")
    if label == -1:
        print("Stray text regions:")
    for line in lines:
        text, confidence = line[1][0], line[1][1]
        viet_text = nom_dict[char2code(text) - 1]
        print(f"Text: {viet_text}, Confidence: {confidence:.4f}")

# Visualize and save results
visualize_results(img, result[0])