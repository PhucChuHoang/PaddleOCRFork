import os
import shutil
from pathlib import Path

def load_unique_chars():
    """Load the unique characters from unique_chars.txt and create an index mapping"""
    with open('unique_chars.txt', 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f.readlines()]
    
    # Create a mapping from character to index
    char_to_index = {char: str(i) for i, char in enumerate(chars)}
    return char_to_index

def process_page_folder(page_folder_path, char_to_index, output_base_dir):
    """Process a single page folder"""
    labels_file = os.path.join(page_folder_path, 'labels.txt')
    
    if not os.path.exists(labels_file):
        print(f"Warning: labels.txt not found in {page_folder_path}")
        return
    
    # Read the labels file
    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Split the line into image path and character
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"Warning: Invalid line format: {line}")
            continue
            
        image_path, character = parts
        
        # Get the filename from the path
        image_filename = os.path.basename(image_path)
        full_image_path = os.path.join(page_folder_path, image_filename)
        
        # Check if the image file exists
        if not os.path.exists(full_image_path):
            print(f"Warning: Image file not found: {full_image_path}")
            continue
        
        # Find the index for this character
        if character not in char_to_index:
            print(f"Warning: Character '{character}' not found in unique_chars.txt")
            continue
        
        char_index = char_to_index[character]
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.join(output_base_dir, char_index)
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the image to the output directory
        output_path = os.path.join(output_dir, image_filename)
        try:
            shutil.copy2(full_image_path, output_path)
            print(f"Copied {image_filename} -> {char_index}/ (character: {character})")
        except Exception as e:
            print(f"Error copying {full_image_path}: {e}")

def main():
    # Load the character to index mapping
    print("Loading unique characters...")
    char_to_index = load_unique_chars()
    print(f"Loaded {len(char_to_index)} unique characters")
    
    # Set up paths
    input_base_dir = os.path.join('Thien-chua-thanh-giao-hoi-toi-kinh', 'cropped')
    output_base_dir = 'organized_by_char_index'
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each page folder
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory {input_base_dir} not found")
        return
    
    page_folders = [f for f in os.listdir(input_base_dir) 
                   if os.path.isdir(os.path.join(input_base_dir, f)) and f.startswith('page_')]
    
    print(f"Found {len(page_folders)} page folders to process")
    
    for page_folder in sorted(page_folders):
        page_folder_path = os.path.join(input_base_dir, page_folder)
        print(f"\nProcessing {page_folder}...")
        process_page_folder(page_folder_path, char_to_index, output_base_dir)
    
    print(f"\nProcessing complete! Images organized in {output_base_dir}/")
    
    # Print summary
    if os.path.exists(output_base_dir):
        index_folders = [f for f in os.listdir(output_base_dir) 
                        if os.path.isdir(os.path.join(output_base_dir, f))]
        print(f"Created {len(index_folders)} character index folders")
        
        # Show a few examples
        print("\nSample folders created:")
        for folder in sorted(index_folders)[:10]:
            folder_path = os.path.join(output_base_dir, folder)
            img_count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
            char_index = int(folder)
            if char_index < len(char_to_index):
                char = list(char_to_index.keys())[char_index]
                print(f"  {folder}/ -> '{char}' ({img_count} images)")

if __name__ == "__main__":
    main() 