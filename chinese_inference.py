from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from contextlib import contextmanager
import threading
import time

DET_MODEL_DIR = 'inference/det/PP-OCRv5_server_det_infer'
REC_MODEL_DIR = 'inference/customized/large/chinese'
REC_CHAR_DICT_PATH = 'ppocr/utils/dict/casia_hwdb_dict.txt'
REC_IMAGE_SHAPE = '3,48,48'
REC_ALGORITHM = 'SVTR'
CONVERT_DICT_PATH = 'chinese_dict.txt'


def getDetectionOcr():
    """Create a PaddleOCR instance for detection only."""
    return PaddleOCR(
        det_model_dir=DET_MODEL_DIR,
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        det=True,
        rec=False,  # Detection only
        cls=False
    )

def getRecognitionOcr():
    """Create a PaddleOCR instance for recognition only."""
    return PaddleOCR(
        rec_model_dir=REC_MODEL_DIR,
        rec_char_dict_path=REC_CHAR_DICT_PATH,
        rec_image_shape=REC_IMAGE_SHAPE,
        rec_algorithm=REC_ALGORITHM,
        use_angle_cls=False,
        use_space_char=True,
        use_gpu=False,
        max_text_length=1,
        drop_score=0,
        show_log=False,
        det=False,  # Recognition only
        rec=True,
        cls=False
    )

class FastOcrProcessor:
    """Fast OCR processor that parallelizes text region recognition within a single image."""
    
    def __init__(self, rec_pool_size=4):
        self.detector = getDetectionOcr()
        self.rec_pool = [getRecognitionOcr() for _ in range(rec_pool_size)]
        self.rec_lock = threading.Lock()
    
    @contextmanager
    def acquire_recognizer(self):
        """Context manager to acquire and release recognition OCR instances safely."""
        recognizer = None
        try:
            with self.rec_lock:
                if self.rec_pool:
                    recognizer = self.rec_pool.pop()
            
            if recognizer is None:
                # If no recognizer available, create a temporary one
                recognizer = getRecognitionOcr()
                yield recognizer
            else:
                yield recognizer
        finally:
            if recognizer is not None:
                with self.rec_lock:
                    # Only return to pool if it was from the original pool
                    if len(self.rec_pool) < 6:  # Prevent unlimited growth
                        self.rec_pool.append(recognizer)
    
    def recognize_text_region(self, img, box):
        """Recognize text in a specific region using the recognition pool."""
        try:
            # Extract the text region from the image
            box = np.array(box).astype(np.int32)
            
            # Handle different box formats
            if box.ndim == 1:
                # If box is 1D, it might be [x1, y1, x2, y2] format
                if len(box) == 4:
                    x_min, y_min, x_max, y_max = box
                else:
                    print(f"Unexpected 1D box format with length {len(box)}")
                    return "", 0.0
            elif box.ndim == 2:
                # If box is 2D, it's in [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                x_min = max(0, int(np.min(box[:, 0])))
                y_min = max(0, int(np.min(box[:, 1])))
                x_max = min(img.shape[1], int(np.max(box[:, 0])))
                y_max = min(img.shape[0], int(np.max(box[:, 1])))
            else:
                print(f"Unexpected box dimensions: {box.ndim}")
                return "", 0.0
            
            # Ensure coordinates are within image bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(img.shape[1], int(x_max))
            y_max = min(img.shape[0], int(y_max))
            
            # Extract region
            text_region = img[y_min:y_max, x_min:x_max]
            
            if text_region.size == 0:
                return "", 0.0
            
            # Use recognition OCR from pool
            with self.acquire_recognizer() as recognizer:
                if recognizer is None:
                    return "", 0.0
                
                # Perform recognition on the text region
                rec_result = recognizer.ocr(text_region, det=False, cls=False)
                
                if rec_result and rec_result[0] and len(rec_result[0]) > 0:
                    text, confidence = rec_result[0][0]
                    return text, confidence
                else:
                    return "", 0.0
                    
        except Exception as e:
            print(f"Error in text region recognition: {e}")
            return "", 0.0
    
    def fast_ocr(self, img, max_workers=4):
        """Perform fast OCR on an image by parallelizing text region recognition."""
        try:
            detection_start = time.time()
            det_result = self.detector.ocr(img, det=True, cls=False, rec=False)
            detection_time = time.time() - detection_start
            
            if not det_result or not det_result[0]:
                print("No text regions detected")
                return []
            
            # Extract detected boxes from detection result
            detected_boxes = det_result[0]
            
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit recognition tasks for all detected regions
                futures = [
                    executor.submit(self.recognize_text_region, img, box) 
                    for box in detected_boxes
                ]
                
                # Collect results with progress bar
                recognition_results = []
                for future in futures:
                    recognition_results.append(future.result())
            
            # Step 3: Combine detection and recognition results
            combined_results = []
            for i, (box, (text, confidence)) in enumerate(zip(detected_boxes, recognition_results)):
                if text:  # Only include if text was recognized
                    combined_results.append([box, (text, confidence)])
            
            return combined_results
            
        except Exception as e:
            print(f"Error in fast OCR processing: {e}")
            return []

# Initialize global fast OCR processor
fast_ocr_processor = FastOcrProcessor(rec_pool_size=4)

def char2code(ch):
    pos = ord(ch) - 0xF0000
    return pos

def load_vietnamese_font(font_size=20):
    try:
        font = ImageFont.truetype("arial", font_size)
        return font
    except Exception:
        print("Arial font not found, using default font")
        return ImageFont.load_default()

def visualize_results(image, result, output_path='visualized_output.jpg'):
    # Load the Vietnamese dictionary
    try:
        with open(CONVERT_DICT_PATH, 'r', encoding='utf-8') as f:
            nom_dict = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: {CONVERT_DICT_PATH} not found!")
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
        
        # Get text and confidence (tuple format)
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

def process_image(image_path, output_path=None, max_workers=4):
    """Process a single image using fast parallel OCR.
    
    Args:
        image_path (str): Path to the image file
        output_path (str, optional): Path for visualization output
        max_workers (int): Number of parallel workers for recognition
    
    Returns:
        dict: Dictionary containing OCR results and metadata
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_name = os.path.basename(image_path)
        
        # Use fast OCR processor
        result = fast_ocr_processor.fast_ocr(img, max_workers=max_workers)        
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = 'output'
            output_filename = f"visualized_{os.path.splitext(image_name)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
        
        # Load the Vietnamese dictionary for label conversion
        try:
            with open(CONVERT_DICT_PATH, 'r', encoding='utf-8') as f:
                nom_dict = f.read().splitlines()
        except FileNotFoundError:
            print(f"Warning: {CONVERT_DICT_PATH} not found! Returning raw results.")
            nom_dict = None
        
        # Convert characters to Vietnamese labels while maintaining PaddleOCR format
        if result and nom_dict:
            converted_result = []
            for item in result:
                box, (text, confidence) = item
                
                # Convert character to Vietnamese text
                try:
                    char_index = char2code(text)
                    if 0 <= char_index < len(nom_dict):
                        viet_text = nom_dict[char_index]
                    else:
                        viet_text = f"[Unknown char: {text}]"
                except Exception as e:
                    viet_text = f"[Error: {text}]"
                
                # Maintain original PaddleOCR format: [box, (text, confidence)]
                converted_item = [box, (viet_text, confidence)]
                converted_result.append(converted_item)
            
            num_regions = len(converted_result)
            print(f"Successfully processed {num_regions} text regions with Vietnamese labels")
            # Wrap in additional list level to match PaddleOCR format: [[[results]]]
            return [converted_result]
        elif result:
            # Return raw result if dictionary not available
            num_regions = len(result)
            print(f"Successfully processed {num_regions} text regions (no label conversion)")
            # Wrap in additional list level to match PaddleOCR format: [[[results]]]
            return [result]
        else:
            print(f"No text found in {image_name}")
            # Return empty result in PaddleOCR format
            return [[]]
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None
        

def process_batch_images(image_paths, max_workers=4, rec_workers=4):
    """Process multiple images using parallel processing.
    
    Args:
        image_paths (list): List of image file paths
        max_workers (int): Maximum number of image processing threads
        rec_workers (int): Number of recognition workers per image
    
    Returns:
        list: List of processing results for each image
    """
    results = []
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_image, img_path, max_workers=rec_workers) 
            for img_path in image_paths
        ]
        
        # Collect results with progress bar
        for future in futures:
            results.append(future.result())
    
    # Print summary
    successful = sum(1 for r in results if r is not None)
    failed = len(results) - successful
    total_regions = sum(len(r[0]) for r in results if r is not None and len(r) > 0)
    
    print(f"\nBatch processing complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Total text regions found: {total_regions}")
    if successful > 0:
        print(f"Average regions per image: {total_regions/successful:.1f}")
    
    return results

if __name__ == "__main__":
    image_path = 'test_images/page_12.png'
    result = process_image(image_path, max_workers=4)
    print(result)