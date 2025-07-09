from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

ocr = PaddleOCR(
    det_model_dir='inference/det/PP-OCRv5_server_det_infer',
    rec_model_dir='inference/customized/svtr_large/27062025/nom',
    rec_char_dict_path='ppocr/utils/dict/new_nom_dict.txt',
    rec_image_shape='3,48,48',
    rec_algorithm='SVTR',
    det_algorithm='DB',
    det_db_thresh=0.2,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.5,
    use_angle_cls=False,
    use_space_char=True,
    use_gpu=True,
    drop_score=0,
)

def char2code(ch):
    pos = ord(ch) - 0xF0000
    return pos

def load_vietnamese_font(font_size=20):
    """Load a font that supports Vietnamese characters with diacritical marks"""
    
    # Test string with Vietnamese characters including diacritics
    vietnamese_test = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    
    # Vietnamese-compatible fonts to try (in order of preference)
    vietnamese_fonts = [
        # Common Vietnamese fonts (Windows)
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/cambria.ttc",
        # Vietnamese specific fonts
        "C:/Windows/Fonts/UVNBachLong_R.TTF",
        "C:/Windows/Fonts/UVNBayBuomHep_R.TTF",
        "C:/Windows/Fonts/VNI-Times.TTF",
        # Google Fonts that support Vietnamese
        "arial.ttf",
        "calibri.ttf", 
        "times.ttf",
        "tahoma.ttf",
        "verdana.ttf",
        "segoeui.ttf",
        # Linux Vietnamese fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        # macOS Vietnamese fonts
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Times.ttc",
        # Project fonts
        os.path.join("doc", "fonts", "simfang.ttf"),
        os.path.join("doc", "fonts", "chinese_cht.ttf"),
    ]
    
    def test_vietnamese_support(font):
        """Test if font properly supports Vietnamese characters"""
        try:
            # Try to create a test image with Vietnamese text
            test_img = Image.new('RGB', (100, 50), 'white')
            test_draw = ImageDraw.Draw(test_img)
            test_draw.text((10, 10), vietnamese_test[:10], font=font, fill='black')
            return True
        except Exception:
            return False
    
    print("Attempting to load Vietnamese-compatible font...")
    
    # Try specific font paths first
    for font_path in vietnamese_fonts:
        try:
            if os.path.exists(font_path) or not font_path.startswith(('/', 'C:')):
                font = ImageFont.truetype(font_path, font_size)
                if test_vietnamese_support(font):
                    print(f"Successfully loaded Vietnamese font: {font_path}")
                    return font
                else:
                    print(f"Font {font_path} loaded but may not fully support Vietnamese diacritics")
                    return font  # Still use it as it's better than default
        except Exception as e:
            continue
    
    # Try common system font names without paths
    font_names = [
        "arial", "Arial", "ARIAL",
        "calibri", "Calibri", "CALIBRI", 
        "times", "Times", "Times New Roman",
        "tahoma", "Tahoma", "TAHOMA",
        "verdana", "Verdana", "VERDANA",
        "segoeui", "Segoe UI",
        "helvetica", "Helvetica",
        "noto", "Noto Sans"
    ]
    
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name, font_size)
            if test_vietnamese_support(font):
                print(f"Successfully loaded Vietnamese system font: {font_name}")
                return font
            else:
                print(f"System font {font_name} loaded but may not fully support Vietnamese diacritics")
                return font  # Still use it as it's better than default
        except Exception:
            continue
    
    # Final fallback - download suggestion
    print("=" * 60)
    print("WARNING: No Vietnamese-compatible fonts found!")
    print("For better Vietnamese text display, please:")
    print("1. Install Google Fonts that support Vietnamese:")
    print("   - Noto Sans Vietnamese")
    print("   - Roboto")
    print("   - Open Sans")
    print("2. Or download Vietnamese fonts like:")
    print("   - VNI-Times")
    print("   - UVN fonts")
    print("3. Place them in your system fonts directory")
    print("=" * 60)
    
    return ImageFont.load_default()

def visualize_results(image, result, output_path='visualized_output.jpg'):
    # Load the Vietnamese dictionary
    try:
        with open('combined_unique_chars.txt', 'r', encoding='utf-8') as f:
            nom_dict = f.read().splitlines()
        print(f"Loaded Vietnamese dictionary with {len(nom_dict)} entries")
    except FileNotFoundError:
        print("Error: combined_unique_chars.txt not found!")
        return
    
    # Convert OpenCV image to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Load Vietnamese-compatible font
    font = load_vietnamese_font(font_size=20)
    
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
        try:
            char_index = char2code(text)
            if 0 <= char_index < len(nom_dict):
                viet_text = nom_dict[char_index]
            else:
                viet_text = f"[Unknown char: {text}]"
                print(f"Warning: Character index {char_index} out of range for dictionary")
        except Exception as e:
            viet_text = f"[Error: {text}]"
            print(f"Error processing character {text}: {e}")
        
        # Display text above the box
        text_position = (int(box[0][0]), int(box[0][1] - 25))  # Moved up a bit more
        label = f"{viet_text} ({confidence:.2f})"
        
        # Get text dimensions for background
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(label, font=font)
        
        # Create semi-transparent background
        overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw semi-transparent background
        padding = 3
        overlay_draw.rectangle(
            [text_position[0] - padding, text_position[1] - padding, 
             text_position[0] + text_width + padding, text_position[1] + text_height + padding],
            fill=(255, 255, 255, 200)  # Semi-transparent white
        )
        
        # Composite the overlay onto the main image
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)  # Recreate draw object
        
        # Draw the text
        draw.text(text_position, label, fill=(255, 0, 0), font=font)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the visualization
    pil_image.save(output_path)
    print(f"Visualization saved to {output_path}")

# Process the image
img_path = 'test_images/page_10.png'
img = cv2.imread(img_path)
result = ocr.ocr(img, det=True, cls=False)

# Print results
print(f"Found {len(result[0])} text regions")

# Load dictionary for printing results
try:
    with open('combined_unique_chars.txt', 'r', encoding='utf-8') as f:
        nom_dict = f.read().splitlines()
    
    for idx, line in enumerate(result[0]):
        text, confidence = line[1][0], line[1][1]
        try:
            char_index = char2code(text)
            if 0 <= char_index < len(nom_dict):
                viet_text = nom_dict[char_index]
            else:
                viet_text = f"[Unknown: {text}]"
        except Exception as e:
            viet_text = f"[Error: {text}]"
        print(f"Text {idx+1}: {viet_text}, Confidence: {confidence:.4f}")
except FileNotFoundError:
    print("Error: combined_unique_chars.txt not found for printing results!")

# Visualize and save results
visualize_results(img, result[0], output_path='output/visualized_vietnamese.png')