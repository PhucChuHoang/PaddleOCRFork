from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

ocr = PaddleOCR(
    det_model_dir='./inference/det/PP-OCRv5_server_det_infer',
    rec_model_dir='./inference/customized/svtr_base',
    rec_char_dict_path=r'./ppocr/utils/dict/casia_hwdb_dict.txt',
    rec_image_shape='3,48,48',
    rec_algorithm='SVTR',
    det_algorithm='DB',
    use_angle_cls=False,
    use_space_char=True,
    use_gpu=True,
    drop_score=0,
)

def char2code(ch):
    pos = ord(ch) - 0xF0000
    return pos

def visualize_results(image, result, output_path='visualized_output.jpg'):
    with open('nom_dict.txt', 'r', encoding='utf-8') as f:
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

# Process the image
img_path = 'test_images/page_102.png'
img = cv2.imread(img_path)
result = ocr.ocr(img, det=True, cls=False)

# Print results
print(f"Found {len(result[0])} text regions")
for idx, line in enumerate(result[0]):
    with open('nom_dict.txt', 'r', encoding='utf-8') as f:
        nom_dict = f.read().splitlines()
    text, confidence = line[1][0], line[1][1]
    viet_text = nom_dict[char2code(text) - 1]
    print(f"Text {idx+1}: {viet_text}, Confidence: {confidence:.4f}")

# Visualize and save results
visualize_results(img, result[0], output_path='output/visualized_page_102.jpg')

