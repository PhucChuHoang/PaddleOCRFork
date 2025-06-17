import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import glob
from pathlib import Path

def setup_font(font_size=16):
    """Setup font for text rendering with fallback options."""
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        return ImageFont.load_default()

def load_alignment_results(alignment_file):
    """Load alignment results from a PaddleOCR format file."""
    try:
        alignment_data = []
        with open(alignment_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse PaddleOCR format: image_path\t[annotations]
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        print(f"Warning: Invalid format in line {line_num}: {line[:50]}...")
                        continue
                    
                    image_path, annotations_str = parts
                    
                    # Parse annotations using eval (safe since we control the format)
                    annotations = eval(annotations_str)
                    
                    # Convert each annotation to our format
                    for annotation in annotations:
                        if len(annotation) >= 2:
                            coordinates = annotation[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            label = annotation[1]  # corrected text
                            
                            # Create alignment data entry
                            alignment_entry = {
                                'coordinates': coordinates,
                                'corrected_label': label,
                                'original_ocr': label,  # In PaddleOCR format, we don't have original OCR
                                'status': 'MATCHED',  # Assume all entries are matched
                                'similarity_score': 1.0,  # Default high similarity
                                'line_index': line_num,
                                'image_path': image_path
                            }
                            alignment_data.append(alignment_entry)
                
                except Exception as e:
                    print(f"Error parsing line {line_num} in {alignment_file}: {e}")
                    continue
        
        return alignment_data
        
    except Exception as e:
        print(f"Error loading {alignment_file}: {e}")
        return []

def parse_coordinates(coordinates):
    """Parse coordinate data and convert to usable format."""
    try:
        # Handle different coordinate formats
        if isinstance(coordinates, list):
            # Check if it's already in the format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if len(coordinates) == 4 and isinstance(coordinates[0], list):
                # Already in correct format, just convert to tuples
                return [(float(point[0]), float(point[1])) for point in coordinates]
            elif len(coordinates) == 4:
                # [x_min, y_min, x_max, y_max] format
                x_min, y_min, x_max, y_max = coordinates
                return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            elif len(coordinates) == 8:
                # [x1, y1, x2, y2, x3, y3, x4, y4] format
                return [(coordinates[i], coordinates[i+1]) for i in range(0, 8, 2)]
        else:
            # String format - remove brackets and split
            coord_str = str(coordinates).strip('[]()').replace(' ', '')
            coords = [float(x) for x in coord_str.split(',')]
            
            if len(coords) == 4:
                # [x_min, y_min, x_max, y_max] format
                x_min, y_min, x_max, y_max = coords
                return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            elif len(coords) == 8:
                # [x1, y1, x2, y2, x3, y3, x4, y4] format
                return [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        
        print(f"Unknown coordinate format: {coordinates}")
        return None
        
    except Exception as e:
        print(f"Error parsing coordinates {coordinates}: {e}")
        return None

def get_status_color(status, similarity=0.0):
    """Get color based on alignment status and similarity."""
    if status == 'MATCHED':
        # For PaddleOCR format, all entries are considered matched
        # Use different shades of green based on line position for variety
        return (0, 255, 0), (0, 150, 0)  # Bright green
    elif status == 'LOW_SIMILARITY':
        if similarity >= 0.3:
            return (255, 165, 0), (200, 100, 0)  # Orange
        else:
            return (255, 255, 0), (200, 200, 0)  # Yellow
    elif status == 'EXTRA_OCR':
        return (255, 0, 255), (200, 0, 200)  # Magenta
    else:  # MISSING_OCR, BOTH_MISSING, or no match
        return (255, 0, 0), (200, 0, 0)  # Red

def visualize_alignment_on_image(image_path, alignment_data, output_path):
    """Visualize alignment results on the original image."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Setup font
        font = setup_font(14)
        
        # Process each aligned result
        for item in alignment_data:
            status = item.get('status', 'UNKNOWN')
            similarity = item.get('similarity_score', 0.0)
            original_ocr = item.get('original_ocr', '')
            corrected_label = item.get('corrected_label', '')
            coordinates = item.get('coordinates', [])
            
            # Parse coordinates
            box_points = parse_coordinates(coordinates)
            if not box_points:
                continue
            
            # Get colors based on status
            box_color, text_color = get_status_color(status, similarity)
            
            # Draw bounding box
            points = [(int(point[0]), int(point[1])) for point in box_points]
            draw.line(points + [points[0]], fill=box_color, width=3)
            
            # Prepare text label
            if status == 'MATCHED':
                # For PaddleOCR format, show the corrected label
                label = f"Text: {corrected_label}"
                label_color = text_color
            elif status == 'LOW_SIMILARITY':
                if original_ocr != corrected_label:
                    label = f"OCR: {original_ocr}\nGT: {corrected_label}\nSim: {similarity:.2f}"
                    label_color = text_color
                else:
                    label = f"Match: {original_ocr}\nSim: {similarity:.2f}"
                    label_color = text_color
            elif status == 'EXTRA_OCR':
                label = f"Extra: {original_ocr}"
                label_color = text_color
            else:
                label = f"No Match: {original_ocr}"
                label_color = text_color
            
            # Position for text (above the box)
            text_x = int(min([p[0] for p in box_points]))
            text_y = int(min([p[1] for p in box_points])) - 25
            
            # Draw text background
            lines = label.split('\n')
            max_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines])
            total_height = len(lines) * 18
            
            # Background rectangle
            draw.rectangle(
                [text_x, text_y, text_x + max_width + 10, text_y + total_height + 5],
                fill=(255, 255, 255, 220)
            )
            
            # Draw text lines
            for i, line in enumerate(lines):
                line_pos = (text_x + 5, text_y + i * 18)
                draw.text(line_pos, line, fill=label_color, font=font)
        
        # Add legend
        legend_y = 10
        legend_items = [
            ("✅ Training Data", (0, 255, 0)),
            ("⚠️ Low Similarity", (255, 165, 0)),
            ("⚠️ Very Low Sim", (255, 255, 0)),
            ("🔴 Extra OCR", (255, 0, 255)),
            ("❌ No Match", (255, 0, 0))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            # Draw color box
            draw.rectangle([10, y_pos, 30, y_pos + 15], fill=color)
            # Draw text
            draw.text((35, y_pos), text, fill=(0, 0, 0), font=font)
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_image.save(output_path)
        print(f"Visualization saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error visualizing {image_path}: {e}")
        return False

def find_alignment_files(aligned_data_folder):
    """Find all alignment PaddleOCR files in the aligned data folder."""
    pattern = os.path.join(aligned_data_folder, '*', '*_training_paddleocr.txt')
    paddle_files = glob.glob(pattern)
    print(f"Found {len(paddle_files)} alignment files")
    return paddle_files

def main():
    """Main function to visualize all aligned data."""
    # Configuration
    images_folder = 'thang10/images'  # Original images folder
    aligned_data_folder = 'aligned_data'  # Aligned data folder
    output_folder = 'alignment_visualizations'  # Output folder for visualizations
    
    print("=== ALIGNMENT VISUALIZATION TOOL ===")
    print(f"Images folder: {images_folder}")
    print(f"Aligned data folder: {aligned_data_folder}")
    print(f"Output folder: {output_folder}")
    
    # Check if folders exist
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' does not exist!")
        return
    
    if not os.path.exists(aligned_data_folder):
        print(f"Error: Aligned data folder '{aligned_data_folder}' does not exist!")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all alignment files
    alignment_files = find_alignment_files(aligned_data_folder)
    
    if not alignment_files:
        print("No alignment files found!")
        return
    
    successful = 0
    failed = 0
    
    # Process each alignment file
    for alignment_file in alignment_files:
        # Extract base name from alignment file path
        base_name = Path(alignment_file).parent.name
        
        # Find corresponding image
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_path = None
        
        for ext in image_extensions:
            potential_path = os.path.join(images_folder, f"{base_name}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            print(f"Warning: No image found for {base_name}")
            failed += 1
            continue
        
        # Load alignment data
        alignment_data = load_alignment_results(alignment_file)
        
        if not alignment_data:
            print(f"Warning: No alignment data in {alignment_file}")
            failed += 1
            continue
        
        # Create visualization
        output_path = os.path.join(output_folder, f"{base_name}_alignment_visualization.jpg")
        
        if visualize_alignment_on_image(image_path, alignment_data, output_path):
            successful += 1
            
            # Print summary for this file
            total_items = len(alignment_data)
            matched = sum(1 for item in alignment_data if item.get('status') == 'MATCHED')
            low_sim = sum(1 for item in alignment_data if item.get('status') == 'LOW_SIMILARITY')
            extra = sum(1 for item in alignment_data if item.get('status') == 'EXTRA_OCR')
            
            print(f"  {base_name}: {total_items} text boxes, {matched} training entries")
        else:
            failed += 1
    
    # Final summary
    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Successfully visualized: {successful}")
    print(f"Failed: {failed}")
    print(f"Total files processed: {successful + failed}")
    print(f"Visualizations saved in: {output_folder}")
    
    if successful > 0:
        print(f"\nTo view the results, check the images in '{output_folder}' folder.")
        print("Legend:")
        print("  ✅ Green: Training data entries (aligned text)")
        print("  ⚠️ Orange/Yellow: Low similarity matches (if any)")
        print("  🔴 Magenta: Extra OCR entries (if any)")
        print("  ❌ Red: Unmatched entries (if any)")

if __name__ == "__main__":
    main() 