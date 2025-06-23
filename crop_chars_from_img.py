import cv2
import os
from PIL import Image
import glob
from pathlib import Path
import ast
from typing import List, Dict
import unicodedata

DICT_PATH = 'nom_dict.txt'
LABEL_FILE = 'labels.txt'
UNIQUE_CHARS_FILE = 'unique_chars.txt'

def load_alignment_results(alignment_file: str):
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
                    annotations = ast.literal_eval(annotations_str)
                    
                    # Convert each annotation to our format
                    for idx, annotation in enumerate(annotations, 1):
                        if len(annotation) >= 2:
                            coordinates = annotation[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            label = annotation[1]  # corrected text

                            coordinates = [(int(coord[0]), int(coord[1])) for coord in coordinates]
                            
                            # Create alignment data entry
                            alignment_entry = {
                                'coordinates': coordinates,
                                'original_ocr': label,
                                'line_index': line_num,
                                'position': idx,
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

def crop_chars_from_image(image_path: str, alignment_data: List[Dict], output_folder: str):
    """Crop character boxes from the original image."""
    try:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        os.makedirs(output_folder, exist_ok=True)
        count = 0

        # Process each aligned result
        with open(os.path.join(output_folder, LABEL_FILE), 'w', encoding='utf-8') as f:
            for item in alignment_data:
                original_ocr = item.get('original_ocr', '')
                coordinates = item.get('coordinates', [])
                line = item.get('line_index', 0)
                position = item.get('position', 0)
                if not coordinates:
                    continue
                
                original_ocr = unicodedata.normalize('NFC', original_ocr)
                # Parse coordinates
                box_points = (coordinates[0][0], coordinates[0][1], coordinates[2][0], coordinates[2][1])
                # Crop the character box
                cropped_image = pil_image.crop(box_points)

                # Save the cropped image
                cropped_filename = f"{image_name}_line_{line}_pos_{position}.png"
                cropped_path = os.path.join(output_folder, cropped_filename)
                cropped_image.save(cropped_path)

                # Write to the text file
                f.write(f"{cropped_path}\t{original_ocr}\n")
                count += 1
        
        print(f"{count} cropped characters saved to {output_folder}")
        return True
        
    except Exception as e:
        print(count)
        print(f"Error processing {image_path}: {e}")
        return False

def update_dict_file(folder_path: str):
    """Update the dictionary file with unique characters from the cropped images."""
    unique_chars = set()
    
    # Read all cropped character files
    for folder in os.listdir(folder_path):
        full_folder_path = os.path.join(folder_path, folder)
        if not os.path.isdir(full_folder_path):
            continue

        label_file_path = os.path.join(full_folder_path, LABEL_FILE)
        if not os.path.exists(label_file_path):
            continue

        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            labels = [line.split('\t')[1] for line in lines if '\t' in line]
            labels = [unicodedata.normalize('NFC', label) for label in labels]
            labels = set(labels)
            unique_chars |= labels
    
    with open(UNIQUE_CHARS_FILE, 'w', encoding='utf-8') as f:
        for char in unique_chars:
            f.write(char + '\n')

    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        char_dict = f.read().splitlines()
        char_dict = [unicodedata.normalize('NFC', char) for char in char_dict]
        existing_chars = set(char_dict)
    
    unique_chars -= existing_chars  # Remove already existing characters
    
    # Append unique characters to the dictionary file
    with open(DICT_PATH, 'a', encoding='utf-8') as f:
        for char in unique_chars:
            f.write(char + '\n')
    
    print(f"Updated dictionary with {len(unique_chars)} unique characters.")

def remap_labels(folder_path: str):
    """Remap labels in the cropped images to fake unicode."""
    # Load the new dictionary
    with open(DICT_PATH, 'r', encoding='utf-8') as dict_file:
        char_dict = dict_file.read().splitlines()
        char_dict = [unicodedata.normalize('NFC', char) for char in char_dict]
    
    # Create a mapping from old labels to new labels
    label_mapping = {label: chr(0xF0000 + i) for i, label in enumerate(char_dict)}
    
    # Process each folder
    for folder in os.listdir(folder_path):
        full_folder_path = os.path.join(folder_path, folder)
        if not os.path.isdir(full_folder_path):
            continue
        
        label_file_path = os.path.join(full_folder_path, LABEL_FILE)
        if not os.path.exists(label_file_path):
            continue

        failed = 0
        
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(label_file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                image_path, original_label = parts[0], parts[1]
                original_label = unicodedata.normalize('NFC', original_label)
                new_label = label_mapping.get(original_label, '#')

                if new_label == '#':
                    failed += 1
                f.write(f"{image_path}\t{new_label}\n")
            
        if failed > 0:
            print(f"Warning: {failed} labels could not be remapped in {label_file_path}.")
    
    print(f"Labels remapped in {folder_path}.")

def find_alignment_files(aligned_data_folder: str):
    """Find all alignment PaddleOCR files in the aligned data folder."""
    pattern = os.path.join(aligned_data_folder, '*', '*_training_paddleocr.txt')
    paddle_files = glob.glob(pattern)
    print(f"Found {len(paddle_files)} alignment files")
    return paddle_files

def main():
    # Configuration
    images_folder = 'thang_12/thang_12'  # Original images folder
    aligned_data_folder = 'thang_12/aligned'  # Aligned data folder
    output_folder = 'thang_12/cropped'  # Output folder for cropped images
    
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
    
    # Process each alignment file
    for alignment_file in alignment_files:
        # Extract base name from alignment file path
        base_name = Path(alignment_file).parent.name
        output_char_folder = os.path.join(output_folder, base_name)
        
        # Find corresponding image
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_path = None
        
        for ext in image_extensions:
            potential_path = os.path.join(images_folder, f"{base_name}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            print(f"No image found for {base_name}")
            continue
        
        # Load alignment data
        alignment_data = load_alignment_results(alignment_file)
        
        if not alignment_data:
            print(f"No alignment data in {alignment_file}")
            continue
        
        if crop_chars_from_image(image_path, alignment_data, output_char_folder):
            
            # Print summary for this file
            total_items = len(alignment_data)            
            print(f"  {base_name}: {total_items} text boxes")
    
    # Update the dictionary file with unique characters
    update_dict_file(output_folder)
    # Remap labels in the cropped images
    # remap_labels(output_folder)
    
    print(f"Cropped images saved in: {output_folder}")

if __name__ == "__main__":
    main() 