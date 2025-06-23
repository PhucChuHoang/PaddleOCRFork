from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from sortedcontainers import SortedDict
import re
import glob
from pathlib import Path

# Configuration constants
DET_MODEL_DIR = 'inference/det/PP-OCRv5_server_det_infer'
REC_MODEL_DIR = 'inference/customized/svtr_base'
REC_CHAR_DICT_PATH = 'ppocr/utils/dict/casia_hwdb_dict.txt'
NOM_DICT_PATH = 'nom_dict.txt'

def initialize_ocr():
    """Initialize and return PaddleOCR instance with predefined configuration."""
    return PaddleOCR(
        det_model_dir=DET_MODEL_DIR,
        rec_model_dir=REC_MODEL_DIR,
        rec_char_dict_path=REC_CHAR_DICT_PATH,
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
    """Convert character to its corresponding code position."""
    pos = ord(ch) - 0xF0000
    return pos

def load_nom_dictionary(dict_path=NOM_DICT_PATH):
    """Load and return the Vietnamese dictionary."""
    with open(dict_path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

def get_vietnamese_text(text, nom_dict):
    """Convert character to Vietnamese text using the dictionary."""
    try:
        return nom_dict[char2code(text) - 1]
    except (IndexError, ValueError):
        return text  # Return original text if conversion fails

def setup_font(font_size=20):
    """Setup font for text rendering with fallback options."""
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        return ImageFont.load_default()

def draw_text_box(draw, box, text, confidence, nom_dict, font):
    """Draw a single text box with Vietnamese translation on the image."""
    # Convert box to integer coordinates
    box = np.array(box).astype(np.int32).reshape(-1, 2)
    
    # Draw polygon around text area
    points = [(point[0], point[1]) for point in box]
    draw.line(points + [points[0]], fill=(0, 255, 0), width=2)
    
    # Get Vietnamese text
    viet_text = get_vietnamese_text(text, nom_dict)
    
    # Prepare label and position
    text_position = (int(box[0][0]), int(box[0][1] - 30))
    label = f"{viet_text} ({confidence:.2f})"
    
    # Add white background for text readability
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = text_bbox[2], text_bbox[3]
    draw.rectangle(
        [text_position[0], text_position[1], 
         text_position[0] + text_width, text_position[1] + text_height],
        fill=(255, 255, 255)
    )
    
    # Draw the text
    draw.text(text_position, label, fill=(255, 0, 0), font=font)

def visualize_results(image, result, output_path='visualized_output.jpg'):
    """Visualize OCR results on the image and save to file."""
    nom_dict = load_nom_dictionary()
    
    # Convert OpenCV image to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Setup font
    font = setup_font()
    
    # Draw detection boxes and recognition results
    for line in result:
        boxes = line[0]
        text, confidence = line[1]
        draw_text_box(draw, boxes, text, confidence, nom_dict, font)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the visualization
    pil_image.save(output_path)
    print(f"Visualization saved to {output_path}")

def cluster_columns(boxes, eps_w_multiplier=0.6, is_vertical=True):
    """
    Group character boxes into columns/rows using 1D clustering.

    Parameters:
        boxes: array-like of shape (N, 4) with [x_min, y_min, x_max, y_max]
        eps_w_multiplier: multiple of median char-width/height for DBSCAN eps
        is_vertical: True for vertical text (cluster by x), False for horizontal (cluster by y)

    Returns:
        labels: integer cluster label per box
    """
    boxes = np.asarray(boxes)
    
    if is_vertical:
        return _cluster_vertical(boxes, eps_w_multiplier)
    else:
        return _cluster_horizontal(boxes, eps_w_multiplier)

def _cluster_vertical(boxes, eps_w_multiplier):
    """Cluster boxes vertically (group by x-coordinates for vertical text)."""
    x_min, _, x_max, _ = boxes.T
    
    # Compute char widths and centroids
    widths = x_max - x_min
    W = np.median(widths)
    x_centroids = (x_min + x_max) / 2
    
    # 1D clustering via DBSCAN
    eps = eps_w_multiplier * W
    clustering = DBSCAN(eps=eps, min_samples=3, n_jobs=-1)
    raw_labels = clustering.fit_predict(x_centroids.reshape(-1, 1))
    
    # Sort clusters by mean-X descending (right-most first)
    return _sort_and_remap_clusters(raw_labels, x_centroids, reverse=True)

def _cluster_horizontal(boxes, eps_w_multiplier):
    """Cluster boxes horizontally (group by y-coordinates for horizontal text)."""
    _, y_min, _, y_max = boxes.T
    
    # Compute char heights and centroids
    heights = y_max - y_min
    H = np.median(heights)
    y_centroids = (y_min + y_max) / 2
    
    # 1D clustering via DBSCAN
    eps = eps_w_multiplier * H
    clustering = DBSCAN(eps=eps, min_samples=2, n_jobs=-1)
    raw_labels = clustering.fit_predict(y_centroids.reshape(-1, 1))
    
    # Sort clusters by mean-Y ascending (top-most first)
    return _sort_and_remap_clusters(raw_labels, y_centroids, reverse=False)

def _sort_and_remap_clusters(raw_labels, centroids, reverse=False):
    """Sort clusters by their centroid positions and remap labels."""
    # Get unique cluster labels (excluding noise = -1)
    unique = [lbl for lbl in np.unique(raw_labels) if lbl >= 0]
    
    # Compute mean centroid of each cluster
    mean_centroids = {lbl: centroids[raw_labels == lbl].mean() for lbl in unique}
    
    # Sort clusters by mean centroid
    sorted_clusters = sorted(unique, key=lambda lbl: mean_centroids[lbl], reverse=reverse)
    
    # Build remapping: old_label → new_label
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    
    # Apply remapping (noise stays -1)
    new_labels = np.array([remap[lbl] if lbl >= 0 else -1 for lbl in raw_labels])
    
    return new_labels

def convert_ocr_result_format(ocr_result):
    """Convert OCR result to simplified format with bounding boxes."""
    altered_result = []
    for line in ocr_result[0]:
        coords = (line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1])
        altered_result.append([coords, line[1]])
    return altered_result

def group_text_by_clusters(text_results, labels, is_vertical=True):
    """Group text results by cluster labels and sort appropriately."""
    clustered_result = SortedDict()
    
    # Group by cluster labels
    for label, line in zip(labels, text_results):
        if label not in clustered_result:
            clustered_result[label] = []
        clustered_result[label].append(line)
    
    # Sort within each cluster
    for label in clustered_result:
        if is_vertical:
            # Sort by y position for vertical text
            clustered_result[label].sort(key=lambda x: x[0][1])
        else:
            # Sort by x position for horizontal text
            clustered_result[label].sort(key=lambda x: x[0][0])
    
    return clustered_result

def process_image(img_path, is_vertical=True, visualize=True):
    """
    Main function to process an image with OCR and clustering.
    
    Parameters:
        img_path: Path to the input image
        is_vertical: True for vertical text layout, False for horizontal
        visualize: Whether to create visualization output
    
    Returns:
        tuple: (original_result, clustered_result)
    """
    # Initialize OCR
    ocr = initialize_ocr()
    
    # Load and process image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Perform OCR
    print("Performing OCR...")
    result = ocr.ocr(img, det=True, cls=False)
    
    # Convert result format
    altered_result = convert_ocr_result_format(result)
    print(f"Found {len(altered_result)} text regions")
    
    # Cluster text regions
    print("Clustering text regions...")
    labels = cluster_columns(
        np.array([line[0] for line in altered_result]), 
        is_vertical=is_vertical
    )
    
    # Group results by clusters
    clustered_result = group_text_by_clusters(altered_result, labels, is_vertical)
    
    # Print results
    # print_clustered_results(clustered_result, is_vertical)
    
    # Visualize if requested
    if visualize:
        visualize_results(img, result[0])
    
    return result[0], clustered_result

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    Parameters:
        s1, s2: Input strings to compare
        
    Returns:
        int: Levenshtein distance between the strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Create a matrix to store distances
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalize_vietnamese_text(text):
    """
    Normalize Vietnamese text by removing punctuation while preserving syllables and diacritical marks.
    Hyphens in compound words are converted to spaces (e.g., "Rô-ma" → "rô ma").
    
    Parameters:
        text: Input Vietnamese text string
        
    Returns:
        str: Normalized Vietnamese text
    """
    if not text:
        return ""
    
    # First, handle hyphens specially - convert them to spaces for compound words
    # This handles cases like "Rô-ma" → "Rô ma", "Giê-ru-sa-lem" → "Giê ru sa lem"
    text_with_spaces = text.replace('-', ' ')
    
    # Remove other punctuation marks but preserve Vietnamese characters
    # This regex removes: underscores, commas, brackets, dots, colons, semicolons, etc.
    # but keeps Vietnamese diacritical marks, letters, and spaces
    punctuation_pattern = r'[_,.\[\](){}:;!?"""''`~@#$%^&*+=|\\/<>]'
    
    # Remove punctuation (excluding hyphens which were already converted to spaces)
    cleaned_text = re.sub(punctuation_pattern, '', text_with_spaces)
    
    # Remove extra whitespace and normalize spacing
    # Split by whitespace and rejoin with single spaces to handle multiple spaces
    words = cleaned_text.split()
    normalized = ' '.join(words)
    
    # Convert to lowercase for matching (Vietnamese diacritics are preserved)
    normalized = normalized.lower()
    
    return normalized.strip()

def normalize_text(text, is_vietnamese=True):
    """
    Normalize text for better matching by removing extra spaces and punctuation.
    
    Parameters:
        text: Input text string
        is_vietnamese: Whether to use Vietnamese-specific normalization
        
    Returns:
        str: Normalized text
    """
    if is_vietnamese:
        return normalize_vietnamese_text(text)
    else:
        # Original normalization for non-Vietnamese text
        return ' '.join(text.lower().strip().split())

def main():
    """Main execution function - runs batch processing automatically."""
    # Default settings for batch processing
    images_folder = 'thang_12/thang_12'  # Default images folder
    texts_folder = 'thang_12/thang_12_txt'  # Default text files folder
    output_folder = 'thang_12/aligned'  # Default output folder
    threshold = 0  # Default similarity threshold
    is_vertical = True  # Default text orientation (vertical)
    debug = False  # Disable debug mode for automatic processing
    
    print("=== VIETNAMESE OCR BATCH TRAINING DATA GENERATOR ===")
    print(f"Processing images from: {images_folder}")
    print(f"Processing text files from: {texts_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Similarity threshold: {threshold}")
    print(f"Text orientation: {'Vertical' if is_vertical else 'Horizontal'}")
    
    # Validate folders
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' does not exist!")
        print("Please create the folder and add your images, or modify the images_folder variable in main()")
        return
    if not os.path.exists(texts_folder):
        print(f"Error: Text files folder '{texts_folder}' does not exist!")
        print("Please create the folder and add your text files, or modify the texts_folder variable in main()")
        return
    
    # Process batch
    batch_results = process_batch_alignment(
        images_folder=images_folder,
        texts_folder=texts_folder,
        output_folder=output_folder,
        threshold=threshold,
        is_vertical=is_vertical,
        debug=debug
    )
    
    if batch_results:
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETED!")
        print(f"{'='*60}")
        print(f"Total files: {batch_results['total_files']}")
        print(f"Successfully processed: {batch_results['processed_files']}")
        print(f"Failed: {batch_results['failed_files']}")
        print(f"Success rate: {batch_results['processed_files']/batch_results['total_files']*100:.1f}%")
        print(f"Total training samples: {batch_results['total_training_samples']}")
        
        print(f"\nResults saved in: {output_folder}")
        print(f"Check batch_summary.txt for detailed results.")
        print(f"Use combined_training_paddleocr.txt for PaddleOCR training.")
    else:
        print("Batch processing failed!")

def levenshtein_distance_words(words1, words2):
    """
    Calculate the Levenshtein distance between two lists of words.
    
    Parameters:
        words1, words2: Lists of words to compare
        
    Returns:
        int: Levenshtein distance between the word sequences
    """
    if len(words1) < len(words2):
        return levenshtein_distance_words(words2, words1)

    if len(words2) == 0:
        return len(words1)

    # Create a matrix to store distances
    previous_row = list(range(len(words2) + 1))
    for i, word1 in enumerate(words1):
        current_row = [i + 1]
        for j, word2 in enumerate(words2):
            # Cost of insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (word1 != word2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def word_level_similarity(text1, text2, is_vietnamese=True):
    """
    Calculate word-level similarity between two texts using word-based Levenshtein distance.
    
    Parameters:
        text1, text2: Input text strings
        is_vietnamese: Whether to use Vietnamese-specific normalization
        
    Returns:
        tuple: (similarity_score, word_distance, details)
    """
    # Normalize texts
    norm_text1 = normalize_text(text1, is_vietnamese)
    norm_text2 = normalize_text(text2, is_vietnamese)
    
    # Split into words
    words1 = norm_text1.split()
    words2 = norm_text2.split()
    
    # Calculate word-level Levenshtein distance
    word_distance = levenshtein_distance_words(words1, words2)
    max_words = max(len(words1), len(words2))
    
    # Convert distance to similarity score (0-1)
    if max_words > 0:
        similarity = 1 - (word_distance / max_words)
    else:
        similarity = 1.0
    
    # Create detailed comparison
    details = {
        'normalized_text1': norm_text1,
        'normalized_text2': norm_text2,
        'words1': words1,
        'words2': words2,
        'word_distance': word_distance,
        'max_words': max_words,
        'similarity': similarity
    }
    
    return similarity, word_distance, details

def combine_cluster_to_sentence(cluster_lines, nom_dict):
    """
    Combine OCR results within a cluster to form a complete sentence.
    
    Parameters:
        cluster_lines: List of OCR results within a cluster
        nom_dict: Vietnamese dictionary for character conversion
        
    Returns:
        tuple: (combined_sentence, word_details, coordinates_list)
    """
    words = []
    word_details = []
    coordinates_list = []
    
    for line in cluster_lines:
        coordinates = line[0]
        text, confidence = line[1]
        
        # Convert to Vietnamese text
        vietnamese_text = get_vietnamese_text(text, nom_dict)
        
        words.append(vietnamese_text)
        word_details.append({
            'coordinates': coordinates,
            'original_text': text,
            'vietnamese_text': vietnamese_text,
            'confidence': confidence
        })
        coordinates_list.append(coordinates)
    
    # Combine words into sentence
    combined_sentence = ' '.join(words)
    
    return combined_sentence, word_details, coordinates_list

def find_sequential_sentence_alignment(clustered_result, reference_texts, threshold=0.6, is_vietnamese=True, debug=False):
    """
    Align OCR sentences with reference sentences in sequential order (1-to-1 mapping).
    Since both are already in reading order, we don't need to search for best matches.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results {cluster_label: [lines]}
        reference_texts: List of reference sentence strings (in order)
        threshold: Similarity threshold for reporting (but all pairs are processed)
        is_vietnamese: Whether to use Vietnamese normalization
        debug: Whether to show debug information
        
    Returns:
        list: Sequential aligned results
    """
    nom_dict = load_nom_dictionary()
    aligned_results = []
    
    # Get sorted cluster labels (columns/rows in order)
    sorted_clusters = sorted(clustered_result.keys())
    
    if debug:
        print(f"\n=== DEBUGGING SEQUENTIAL SENTENCE ALIGNMENT ===")
        print(f"Threshold: {threshold} (for reporting only)")
        print(f"Number of OCR lines: {len(sorted_clusters)}")
        print(f"Number of reference sentences: {len(reference_texts)}")
        print(f"Alignment: 1-to-1 sequential mapping")
        print("\nOCR lines:")
        for i, cluster_label in enumerate(sorted_clusters[:3]):
            lines_count = len(clustered_result[cluster_label])
            print(f"  Line {i+1} (Cluster {cluster_label}): {lines_count} words")
        if len(sorted_clusters) > 3:
            print(f"  ... and {len(sorted_clusters) - 3} more lines")
        
        print("\nReference sentences:")
        for i, ref_text in enumerate(reference_texts[:3]):
            words = normalize_text(ref_text, is_vietnamese).split()
            print(f"  Ref {i+1}: '{ref_text[:60]}...' ({len(words)} words)")
        if len(reference_texts) > 3:
            print(f"  ... and {len(reference_texts) - 3} more")
    
    # Sequential alignment: OCR line i → Reference line i
    max_pairs = max(len(sorted_clusters), len(reference_texts))
    
    for i in range(max_pairs):
        if debug and i < 5:  # Debug first few pairs
            print(f"\n--- Aligning Pair {i+1} ---")
        
        # Get OCR line (if exists)
        if i < len(sorted_clusters):
            cluster_label = sorted_clusters[i]
            cluster_lines = clustered_result[cluster_label]
            ocr_sentence, word_details, coordinates_list = combine_cluster_to_sentence(cluster_lines, nom_dict)
        else:
            # No more OCR lines
            cluster_label = f"missing_{i}"
            ocr_sentence = ""
            word_details = []
            coordinates_list = []
        
        # Get reference sentence (if exists)
        if i < len(reference_texts):
            reference_sentence = reference_texts[i]
        else:
            # No more reference sentences
            reference_sentence = ""
        
        if debug and i < 5:
            print(f"OCR Line {i+1}: '{ocr_sentence}'")
            print(f"Ref Line {i+1}: '{reference_sentence[:60]}...'")
        
        # Calculate similarity if both exist
        if ocr_sentence and reference_sentence:
            similarity, word_distance, details = word_level_similarity(ocr_sentence, reference_sentence, is_vietnamese)
            
            if debug and i < 3:
                print(f"OCR words: {details['words1'][:8]}...")  # Show first 8 words
                print(f"Ref words: {details['words2'][:8]}...")  # Show first 8 words
                print(f"Word distance: {word_distance}, Max words: {details['max_words']}")
                print(f"Similarity: {similarity:.3f}")
                
                # Show word-by-word alignment for first few
                print(f"Word alignment (first 10 words):")
                words1, words2 = details['words1'][:10], details['words2'][:10]
                max_len = max(len(words1), len(words2))
                for j in range(min(max_len, 10)):
                    w1 = words1[j] if j < len(words1) else "∅"
                    w2 = words2[j] if j < len(words2) else "∅"
                    match = "✓" if w1 == w2 else "✗"
                    print(f"  {j:2d}: '{w1}' vs '{w2}' {match}")
            
            # Determine status based on threshold
            status = "MATCHED" if similarity >= threshold else "LOW_SIMILARITY"
            
        elif ocr_sentence and not reference_sentence:
            similarity = 0.0
            status = "EXTRA_OCR"
        elif not ocr_sentence and reference_sentence:
            similarity = 0.0
            status = "MISSING_OCR"
        else:
            similarity = 0.0
            status = "BOTH_MISSING"
        
        if debug and i < 3:
            print(f"Status: {status} (similarity: {similarity:.3f})")
        
        # Add to results
        aligned_results.append({
            'pair_index': i + 1,
            'cluster_label': cluster_label,
            'ocr_sentence': ocr_sentence,
            'reference_sentence': reference_sentence,
            'similarity': similarity,
            'word_details': word_details,
            'coordinates_list': coordinates_list,
            'status': status
        })
    
    if debug:
        matched_count = sum(1 for result in aligned_results if result['status'] == 'MATCHED')
        low_sim_count = sum(1 for result in aligned_results if result['status'] == 'LOW_SIMILARITY')
        print(f"\n=== SEQUENTIAL ALIGNMENT SUMMARY ===")
        print(f"Total pairs processed: {len(aligned_results)}")
        print(f"High similarity (≥{threshold}): {matched_count}")
        print(f"Low similarity (<{threshold}): {low_sim_count}")
        print(f"Extra OCR lines: {sum(1 for r in aligned_results if r['status'] == 'EXTRA_OCR')}")
        print(f"Missing OCR lines: {sum(1 for r in aligned_results if r['status'] == 'MISSING_OCR')}")
    
    return aligned_results

def extract_all_words_with_coordinates(clustered_result, nom_dict):
    """
    Extract all words from clustered OCR results in reading order with their coordinates.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        nom_dict: Vietnamese dictionary for text conversion
        
    Returns:
        list: List of (word, word_detail) tuples in reading order
    """
    all_words_with_coords = []
    
    # Process clusters in order (columns/rows)
    sorted_clusters = sorted(clustered_result.keys())
    
    for cluster_label in sorted_clusters:
        cluster_lines = clustered_result[cluster_label]
        
        # Process words within each cluster
        for line in cluster_lines:
            coordinates = line[0]
            text, confidence = line[1]
            
            # Convert to Vietnamese text
            vietnamese_text = get_vietnamese_text(text, nom_dict)
            
            word_detail = {
                'coordinates': coordinates,
                'original_text': text,
                'vietnamese_text': vietnamese_text,
                'confidence': confidence,
                'cluster_label': cluster_label
            }
            
            all_words_with_coords.append((vietnamese_text, word_detail))
    
    return all_words_with_coords

def process_sequential_sentence_alignment(img_path, reference_texts, threshold=0.5, is_vertical=True, visualize=True, debug=False, use_anchors=True):
    """
    Main function with word-level anchor-based alignment as requested by user.
    
    Parameters:
        img_path: Path to input image
        reference_texts: List of reference sentence strings (in order)
        threshold: Similarity threshold for reporting quality
        is_vertical: Text layout orientation  
        visualize: Whether to create visualizations
        debug: Whether to show debug information
        use_anchors: Whether to use word-level anchor-based alignment (recommended)
        
    Returns:
        tuple: (original_ocr_results, clustered_results, aligned_results)
    """
    # Perform OCR and clustering first
    print("Processing image with OCR and clustering...")
    original_result, clustered_result = process_image(img_path, is_vertical, visualize=False)
    
    print(f"\nFound {len(clustered_result)} {'columns' if is_vertical else 'rows'} (OCR lines)")
    
    if use_anchors:
        print("Using WORD-LEVEL anchor-based alignment as requested...")
        
        # Load Vietnamese dictionary
        nom_dict = load_nom_dictionary()
        
        # Extract all words with coordinates in reading order 
        ocr_words_with_coords = extract_all_words_with_coordinates(clustered_result, nom_dict)
        
        # Combine all reference texts into one text for word-level processing
        combined_reference_text = ' '.join(reference_texts)
        print(f"Combined reference text: {len(combined_reference_text.split())} words")
        
        # Perform word-level anchor-guided alignment
        aligned_results = word_level_anchor_guided_alignment(
            ocr_words_with_coords, 
            combined_reference_text,
            min_anchor_similarity=0.9,  # High threshold for exact word matches
            threshold=threshold, 
            is_vietnamese=True,
            debug=debug
        )
    else:
        print("Using sequential sentence-level alignment...")
        print(f"Similarity threshold: {threshold} (for quality reporting)")
        
        aligned_results = find_sequential_sentence_alignment(
            clustered_result, 
            reference_texts, 
            threshold=threshold, 
            is_vietnamese=True,
            debug=debug
        )
    
    # Create visualizations if requested
    if visualize:
        img = cv2.imread(img_path)
        visualize_results(img, original_result, 'original_ocr_visualization.jpg')
        
        # Create alignment visualization
        if use_anchors:
            visualize_word_level_alignment(img, aligned_results, 'word_level_alignment_visualization.jpg')
        else:
            visualize_sequential_alignment(img, aligned_results, 'alignment_visualization.jpg')
    
    return original_result, clustered_result, aligned_results

def visualize_sequential_alignment(image, aligned_results, output_path='sequential_alignment_visualization.jpg'):
    """
    Visualize sequential alignment results with color coding for different statuses.
    """
    # Convert OpenCV image to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Setup font
    font = setup_font(14)
    
    for result in aligned_results:
        pair_index = result['pair_index']
        word_details = result['word_details']
        status = result['status']
        similarity = result['similarity']
        
        # Color coding based on status
        if status == 'MATCHED':
            box_color = (0, 255, 0)  # Green for good matches
            text_color = (0, 150, 0)
        elif status == 'LOW_SIMILARITY':
            box_color = (255, 165, 0)  # Orange for low similarity
            text_color = (200, 100, 0)
        elif status == 'EXTRA_OCR':
            box_color = (255, 255, 0)  # Yellow for extra OCR
            text_color = (200, 200, 0)
        else:  # MISSING_OCR, BOTH_MISSING
            box_color = (255, 0, 0)  # Red for missing
            text_color = (200, 0, 0)
        
        # Draw boxes around each word if OCR exists
        if word_details:
            for word_detail in word_details:
                coordinates = word_detail['coordinates']
                
                # Convert coordinates to box format if needed
                if len(coordinates) == 4 and not isinstance(coordinates[0], (int, float)):
                    x_coords = [point[0] for point in coordinates]
                    y_coords = [point[1] for point in coordinates]
                    box = np.array([[min(x_coords), min(y_coords)], [max(x_coords), min(y_coords)], 
                                   [max(x_coords), max(y_coords)], [min(x_coords), max(y_coords)]])
                else:
                    x_min, y_min, x_max, y_max = coordinates
                    box = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                
                # Draw bounding box
                points = [(point[0], point[1]) for point in box]
                draw.line(points + [points[0]], fill=box_color, width=2)
            
            # Add pair label
            first_coords = word_details[0]['coordinates']
            if len(first_coords) == 4 and not isinstance(first_coords[0], (int, float)):
                x_coords = [point[0] for point in first_coords]
                y_coords = [point[1] for point in first_coords]
                label_pos = (min(x_coords), min(y_coords) - 40)
            else:
                label_pos = (first_coords[0], first_coords[1] - 40)
            
            # Create status label
            if status == 'MATCHED':
                label = f"Pair {pair_index}: ✅ MATCH ({similarity:.2f})"
            elif status == 'LOW_SIMILARITY':
                label = f"Pair {pair_index}: ⚠️ LOW ({similarity:.2f})"
            elif status == 'EXTRA_OCR':
                label = f"Pair {pair_index}: ⚠️ EXTRA"
            else:
                label = f"Pair {pair_index}: ❌ MISSING"
            
            # Draw label background and text
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width, text_height = text_bbox[2], text_bbox[3]
            draw.rectangle(
                [label_pos[0], label_pos[1], label_pos[0] + text_width + 10, label_pos[1] + text_height + 5],
                fill=(255, 255, 255, 200)
            )
            
            draw.text((label_pos[0] + 5, label_pos[1]), label, fill=text_color, font=font)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save visualization
    pil_image.save(output_path)
    print(f"Sequential alignment visualization saved to {output_path}")

def generate_training_data_from_alignment(aligned_results):
    """
    Generate training data from word-level alignment results.
    
    Parameters:
        aligned_results: List of word-level alignment results
        
    Returns:
        list: Training data entries with coordinates and corrected labels
    """
    training_data = []
    
    for result in aligned_results:
        pair_index = result['pair_index']
        cluster_label = result['cluster_label']
        ocr_word = result.get('ocr_word', '')
        reference_word = result.get('reference_word', '')
        similarity = result['similarity']
        word_detail = result.get('word_detail')
        status = result['status']
        alignment_type = result.get('alignment_type', 'UNKNOWN')
        
        # Only process entries that have OCR word and word detail
        if word_detail and ocr_word:
            coordinates = word_detail['coordinates']
            original_text = word_detail['vietnamese_text']
            confidence = word_detail['confidence']
            
            # Determine corrected label based on alignment
            if status in ['MATCHED', 'LOW_SIMILARITY'] and reference_word:
                # Use reference word as corrected label
                corrected_label = reference_word
            elif status == 'EXTRA_OCR':
                # Keep original OCR text for extra words
                corrected_label = original_text
            else:
                # Keep original for other cases
                corrected_label = original_text
            
            training_entry = {
                'coordinates': coordinates,
                'original_ocr': original_text,
                'corrected_label': corrected_label,
                'confidence': confidence,
                'word_index': pair_index,
                'cluster_label': cluster_label,
                'similarity_score': similarity,
                'status': status,
                'alignment_type': alignment_type
            }
            
            training_data.append(training_entry)
    
    return training_data

def save_training_data(training_data, output_file='training_data.txt', format_type='paddleocr', actual_image_path=None):
    """
    Save training data in various formats for semi-supervised learning.
    
    Parameters:
        training_data: List of training data entries
        output_file: Output file path
        format_type: Format type ('paddleocr', 'json', 'simple')
        actual_image_path: Actual image path for PaddleOCR format
    """
    try:
        if format_type == 'paddleocr':
            save_paddleocr_format(training_data, output_file, actual_image_path)
        elif format_type == 'simple':
            save_simple_format(training_data, output_file)
        else:
            print(f"Unknown format type: {format_type}")
            return
            
        print(f"Training data saved to {output_file} in {format_type} format")
        print(f"Total training samples: {len(training_data)}")
        
        # Print statistics
        matched_count = sum(1 for entry in training_data if entry['status'] in ['MATCHED', 'LOW_SIMILARITY'])
        corrected_count = sum(1 for entry in training_data if entry['original_ocr'] != entry['corrected_label'])
        
        print(f"Statistics:")
        print(f"  - Entries from matched lines: {matched_count}")
        print(f"  - Entries with corrections: {corrected_count}")
        print(f"  - Correction rate: {corrected_count/len(training_data)*100:.1f}%")
        
    except Exception as e:
        print(f"Error saving training data: {e}")

def save_paddleocr_format(training_data, output_file, actual_image_path=None):
    """Save in PaddleOCR training format: image_path\t[coordinates, label]"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Group by cluster for better organization
        current_cluster = None
        cluster_entries = []
        
        for entry in training_data:
            cluster_label = entry['cluster_label']
            
            if current_cluster != cluster_label:
                # Write previous cluster if exists
                if cluster_entries:
                    write_paddleocr_line(f, current_cluster, cluster_entries, actual_image_path)
                
                # Start new cluster
                current_cluster = cluster_label
                cluster_entries = [entry]
            else:
                cluster_entries.append(entry)
        
        # Write last cluster
        if cluster_entries:
            write_paddleocr_line(f, current_cluster, cluster_entries, actual_image_path)

def write_paddleocr_line(f, cluster_label, entries, actual_image_path=None):
    """Write a single cluster in PaddleOCR format"""
    # Use actual image path if provided, otherwise use dummy path
    if actual_image_path:
        image_path = os.path.basename(actual_image_path)
    else:
        image_path = f"cluster_{cluster_label}.jpg"
    
    annotations = []
    for entry in entries:
        coords = entry['coordinates']
        label = entry['corrected_label']
        
        # Convert coordinates to PaddleOCR format
        if len(coords) == 4 and not isinstance(coords[0], (int, float)):
            # Convert from [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
            points = [[int(point[0]), int(point[1])] for point in coords]
        else:
            # Convert from [x_min, y_min, x_max, y_max] to 4 corners
            x_min, y_min, x_max, y_max = coords
            points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        
        annotation = [points, label]
        annotations.append(annotation)
    
    # Write in PaddleOCR format: image_path\t[annotation1, annotation2, ...]
    f.write(f"{image_path}\t{annotations}\n")

def save_simple_format(training_data, output_file):
    """Save in simple text format for easy inspection"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WORD-LEVEL TRAINING DATA FOR SEMI-SUPERVISED LEARNING\n")
        f.write("=" * 60 + "\n\n")
        
        current_cluster = None
        for entry in training_data:
            cluster_label = entry['cluster_label']
            
            if current_cluster != cluster_label:
                f.write(f"\n=== Cluster {cluster_label} ===\n")
                current_cluster = cluster_label
            
            coords = entry['coordinates']
            original = entry['original_ocr']
            corrected = entry['corrected_label']
            confidence = entry['confidence']
            status = entry['status']
            alignment_type = entry.get('alignment_type', 'UNKNOWN')
            word_index = entry['word_index']
            
            f.write(f"Word #{word_index}: {alignment_type} alignment\n")
            f.write(f"Coordinates: {coords}\n")
            f.write(f"Original OCR: '{original}'\n")
            f.write(f"Corrected Label: '{corrected}'\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"Status: {status}\n")
            
            if original != corrected:
                f.write(f"*** CORRECTION APPLIED ***\n")
            
            f.write("-" * 40 + "\n")

def load_ground_truth_file(txt_path):
    """
    Load ground truth text from a file and split into sentences/lines.
    
    Parameters:
        txt_path: Path to the ground truth text file
        
    Returns:
        list: List of ground truth sentences/lines
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by lines and filter out empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        return lines
    except Exception as e:
        print(f"Error loading ground truth file {txt_path}: {e}")
        return []

def find_matching_files(images_folder, texts_folder):
    """
    Find matching image and text file pairs based on filename.
    
    Parameters:
        images_folder: Path to folder containing images
        texts_folder: Path to folder containing text files
        
    Returns:
        list: List of (image_path, text_path, base_name) tuples
    """
    matching_pairs = []
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
    
    print(f"Found {len(image_files)} image files in {images_folder}")
    
    # Find matching text files
    for image_path in image_files:
        base_name = Path(image_path).stem  # Get filename without extension
        
        # Look for corresponding text file
        txt_path = os.path.join(texts_folder, f"{base_name}.txt")
        
        if os.path.exists(txt_path):
            matching_pairs.append((image_path, txt_path, base_name))
        else:
            print(f"Warning: No matching text file found for {base_name}")
    
    print(f"Found {len(matching_pairs)} matching image-text pairs")
    return matching_pairs

def process_batch_alignment(images_folder, texts_folder, output_folder='batch_results', 
                          threshold=0.4, is_vertical=True, debug=False):
    """
    Process a batch of images and corresponding text files with automatic alignment.
    
    Parameters:
        images_folder: Path to folder containing images
        texts_folder: Path to folder containing ground truth text files
        output_folder: Path to output folder for results
        threshold: Similarity threshold for alignment quality
        is_vertical: Text layout orientation
        debug: Whether to show debug information
        
    Returns:
        dict: Batch processing results
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find matching image-text pairs
    matching_pairs = find_matching_files(images_folder, texts_folder)
    
    if not matching_pairs:
        print("No matching image-text pairs found!")
        return None
    
    # Initialize batch results
    batch_results = {
        'total_files': len(matching_pairs),
        'processed_files': 0,
        'failed_files': 0,
        'total_training_samples': 0,
        'file_results': []
    }
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(matching_pairs)} IMAGE-TEXT PAIRS")
    print(f"{'='*80}")
    
    # Process each pair
    for i, (image_path, txt_path, base_name) in enumerate(matching_pairs, 1):
        print(f"\n--- Processing {i}/{len(matching_pairs)}: {base_name} ---")
        
        try:
            # Load ground truth
            reference_texts = load_ground_truth_file(txt_path)
            
            if not reference_texts:
                print(f"Skipping {base_name}: No ground truth text found")
                batch_results['failed_files'] += 1
                continue
            
            print(f"Loaded {len(reference_texts)} reference lines from {base_name}.txt")
            
            # Process image with anchor-based alignment
            original_result, clustered_result, aligned_results = process_sequential_sentence_alignment(
                img_path=image_path,
                reference_texts=reference_texts,
                threshold=threshold,
                is_vertical=is_vertical,
                visualize=False,  # Skip visualization for batch processing
                debug=debug and i <= 3,  # Only debug first 3 files
                use_anchors=True  # Enable anchor-based alignment
            )
            
            # Generate training data
            training_data = generate_training_data_from_alignment(aligned_results)
            
            # Save results for this file
            file_output_folder = os.path.join(output_folder, base_name)
            os.makedirs(file_output_folder, exist_ok=True)
            
            # Save training data in PaddleOCR format
            save_training_data(training_data, 
                             os.path.join(file_output_folder, f'{base_name}_training_paddleocr.txt'), 
                             'paddleocr', image_path)
            
            # Save alignment results for reference
            save_word_level_alignment_results(aligned_results, 
                                             os.path.join(file_output_folder, f'{base_name}_alignment.txt'))
            
            # Update batch statistics
            batch_results['processed_files'] += 1
            batch_results['total_training_samples'] += len(training_data)
            
            # Calculate file-level statistics
            matched_count = sum(1 for result in aligned_results if result['status'] == 'MATCHED')
            low_sim_count = sum(1 for result in aligned_results if result['status'] == 'LOW_SIMILARITY')
            
            file_result = {
                'base_name': base_name,
                'ocr_lines': len(clustered_result),
                'reference_lines': len(reference_texts),
                'training_samples': len(training_data),
                'matched_lines': matched_count,
                'low_similarity_lines': low_sim_count,
                'match_rate': matched_count / len(aligned_results) * 100 if aligned_results else 0
            }
            
            batch_results['file_results'].append(file_result)
            
            print(f"✅ {base_name}: {len(training_data)} training samples, {matched_count} good matches")
            
        except Exception as e:
            print(f"❌ Error processing {base_name}: {e}")
            batch_results['failed_files'] += 1
            
            if debug:
                import traceback
                traceback.print_exc()
    
    # Save batch summary
    save_batch_summary(batch_results, output_folder)
    
    return batch_results

def save_batch_summary(batch_results, output_folder):
    """
    Save a summary of batch processing results.
    
    Parameters:
        batch_results: Dictionary containing batch processing results
        output_folder: Output folder path
    """
    summary_path = os.path.join(output_folder, 'batch_summary.txt')
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write(f"Total files: {batch_results['total_files']}\n")
            f.write(f"Successfully processed: {batch_results['processed_files']}\n")
            f.write(f"Failed: {batch_results['failed_files']}\n")
            f.write(f"Success rate: {batch_results['processed_files']/batch_results['total_files']*100:.1f}%\n")
            f.write(f"Total training samples generated: {batch_results['total_training_samples']}\n\n")
            
            # Per-file results
            f.write("PER-FILE RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for result in batch_results['file_results']:
                f.write(f"\nFile: {result['base_name']}\n")
                f.write(f"  OCR lines: {result['ocr_lines']}\n")
                f.write(f"  Reference lines: {result['reference_lines']}\n")
                f.write(f"  Training samples: {result['training_samples']}\n")
                f.write(f"  Good matches: {result['matched_lines']}\n")
                f.write(f"  Low similarity: {result['low_similarity_lines']}\n")
                f.write(f"  Match rate: {result['match_rate']:.1f}%\n")
        
        print(f"\nBatch summary saved to {summary_path}")
        
    except Exception as e:
        print(f"Error saving batch summary: {e}")

def word_level_anchor_guided_alignment(ocr_words_with_coords, reference_text, min_anchor_similarity=0.9, threshold=0.4, is_vietnamese=True, debug=False):
    """
    Perform word-level alignment using dynamic programming approach similar to Levenshtein distance.
    Minimizes edit operations (insert, delete, substitute) while using anchor points as low-cost alignments.
    
    Example:
    OCR words: [A, B, C, D, E, F, G, H] 
    Reference: [A, B, M, N, E, F, G, L]
    
    DP finds optimal path minimizing:
    - Substitute C→M, D→N, H→L (cost based on similarity)  
    - Keep A→A, B→B, E→E, F→F, G→G as low-cost anchors
    
    Parameters:
        ocr_words_with_coords: List of (word, word_detail) tuples from OCR
        reference_text: Single text string containing all reference words
        min_anchor_similarity: Minimum similarity for considering as anchor (low cost)
        threshold: Similarity threshold for reporting quality
        is_vietnamese: Whether to use Vietnamese normalization
        debug: Whether to show debug information
        
    Returns:
        list: Optimally aligned results minimizing edit distance
    """
    # Split reference text into words
    reference_words = normalize_text(reference_text, is_vietnamese).split()
    
    if debug:
        print(f"\n=== LEVENSHTEIN-STYLE WORD ALIGNMENT ===")
        print(f"OCR words: {len(ocr_words_with_coords)}")
        print(f"Reference words: {len(reference_words)}")
    
    # Extract OCR words for easier processing
    ocr_words = [normalize_text(word, is_vietnamese) for word, _ in ocr_words_with_coords]
    
    if debug:
        print(f"OCR sequence: {ocr_words[:10]}..." if len(ocr_words) > 10 else f"OCR sequence: {ocr_words}")
        print(f"Ref sequence: {reference_words[:10]}..." if len(reference_words) > 10 else f"Ref sequence: {reference_words}")
    
    # Perform DP alignment
    alignment_path, total_cost = dp_word_alignment(
        ocr_words, 
        reference_words, 
        min_anchor_similarity=min_anchor_similarity,
        debug=debug
    )
    
    if debug:
        print(f"Optimal alignment found with total cost: {total_cost:.3f}")
        print(f"Alignment path length: {len(alignment_path)}")
    
    # Convert alignment path to results
    aligned_results = convert_alignment_path_to_results(
        alignment_path, 
        ocr_words_with_coords, 
        reference_words, 
        threshold,
        is_vietnamese,
        debug
    )
    
    if debug:
        matched_count = sum(1 for r in aligned_results if r['status'] == 'MATCHED')
        anchor_count = sum(1 for r in aligned_results if r.get('is_anchor', False))
        print(f"\n=== DP ALIGNMENT SUMMARY ===")
        print(f"Total operations: {len(aligned_results)}")
        print(f"Anchor alignments: {anchor_count}")
        print(f"High similarity matches: {matched_count}")
        print(f"Edit distance cost: {total_cost:.3f}")
    
    return aligned_results

def dp_word_alignment(ocr_words, reference_words, min_anchor_similarity=0.9, debug=False):
    """
    Dynamic programming word alignment minimizing edit distance with anchor point bonuses.
    
    Parameters:
        ocr_words: List of normalized OCR words
        reference_words: List of normalized reference words  
        min_anchor_similarity: Threshold for anchor point detection
        debug: Whether to show debug information
        
    Returns:
        tuple: (alignment_path, total_cost)
            alignment_path: List of (operation, ocr_idx, ref_idx, cost) tuples
            total_cost: Total alignment cost
    """
    m, n = len(ocr_words), len(reference_words)
    
    # Define operation costs
    INSERT_COST = 1.0      # Cost to insert a reference word (missing OCR)
    DELETE_COST = 1.0      # Cost to delete an OCR word (extra OCR)
    ANCHOR_COST = 0.1      # Very low cost for anchor matches
    
    def substitution_cost(ocr_word, ref_word):
        """Calculate substitution cost based on word similarity"""
        if not ocr_word or not ref_word:
            return 1.0
            
        if ocr_word == ref_word:
            return ANCHOR_COST  # Exact match = anchor
            
        # Calculate similarity
        max_len = max(len(ocr_word), len(ref_word))
        if max_len == 0:
            return ANCHOR_COST
            
        edit_dist = levenshtein_distance(ocr_word, ref_word)
        similarity = 1 - (edit_dist / max_len)
        
        if similarity >= min_anchor_similarity:
            return ANCHOR_COST  # High similarity = anchor
        else:
            return 1.0 - similarity  # Cost inversely proportional to similarity
    
    # Initialize DP table
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    parent = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    dp[0][0] = 0
    
    # Initialize first row (all insertions)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + INSERT_COST
        parent[0][j] = ('INSERT', 0, j-1)
    
    # Initialize first column (all deletions)  
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + DELETE_COST
        parent[i][0] = ('DELETE', i-1, 0)
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ocr_word = ocr_words[i-1]
            ref_word = reference_words[j-1]
            
            # Option 1: Substitute (or match)
            sub_cost = substitution_cost(ocr_word, ref_word)
            if dp[i-1][j-1] + sub_cost < dp[i][j]:
                dp[i][j] = dp[i-1][j-1] + sub_cost
                parent[i][j] = ('SUBSTITUTE', i-1, j-1)
            
            # Option 2: Delete OCR word
            if dp[i-1][j] + DELETE_COST < dp[i][j]:
                dp[i][j] = dp[i-1][j] + DELETE_COST
                parent[i][j] = ('DELETE', i-1, j)
            
            # Option 3: Insert reference word
            if dp[i][j-1] + INSERT_COST < dp[i][j]:
                dp[i][j] = dp[i][j-1] + INSERT_COST
                parent[i][j] = ('INSERT', i, j-1)
    
    # Backtrack to find optimal path
    alignment_path = []
    i, j = m, n
    total_cost = dp[m][n]
    
    while i > 0 or j > 0:
        if parent[i][j] is None:
            break
            
        operation, prev_i, prev_j = parent[i][j]
        
        if operation == 'SUBSTITUTE':
            ocr_word = ocr_words[i-1]
            ref_word = reference_words[j-1]
            cost = substitution_cost(ocr_word, ref_word)
            alignment_path.append((operation, i-1, j-1, cost))
            i, j = prev_i, prev_j
            
        elif operation == 'DELETE':
            alignment_path.append((operation, i-1, -1, DELETE_COST))
            i = prev_i
            
        elif operation == 'INSERT':
            alignment_path.append((operation, -1, j-1, INSERT_COST))
            j = prev_j
    
    # Reverse to get forward order
    alignment_path.reverse()
    
    if debug:
        print(f"\nDP alignment path:")
        for i, (op, ocr_idx, ref_idx, cost) in enumerate(alignment_path[:10]):  # Show first 10
            if op == 'SUBSTITUTE':
                ocr_w = ocr_words[ocr_idx] if ocr_idx >= 0 else "?"
                ref_w = reference_words[ref_idx] if ref_idx >= 0 else "?"
                anchor_marker = "🔗" if cost <= ANCHOR_COST + 0.01 else ""
                print(f"  {i+1}: {op} '{ocr_w}' → '{ref_w}' (cost: {cost:.2f}) {anchor_marker}")
            elif op == 'DELETE':
                ocr_w = ocr_words[ocr_idx] if ocr_idx >= 0 else "?"
                print(f"  {i+1}: {op} '{ocr_w}' (cost: {cost:.2f})")
            elif op == 'INSERT':
                ref_w = reference_words[ref_idx] if ref_idx >= 0 else "?"
                print(f"  {i+1}: {op} → '{ref_w}' (cost: {cost:.2f})")
        
        if len(alignment_path) > 10:
            print(f"  ... and {len(alignment_path) - 10} more operations")
    
    return alignment_path, total_cost

def convert_alignment_path_to_results(alignment_path, ocr_words_with_coords, reference_words, threshold, is_vietnamese, debug=False):
    """
    Convert DP alignment path to structured results format.
    
    Parameters:
        alignment_path: List of (operation, ocr_idx, ref_idx, cost) tuples
        ocr_words_with_coords: Original OCR words with coordinates
        reference_words: Reference word list
        threshold: Similarity threshold for status classification
        is_vietnamese: Whether using Vietnamese normalization
        debug: Whether to show debug information
        
    Returns:
        list: Structured alignment results
    """
    aligned_results = []
    ANCHOR_COST_THRESHOLD = 0.15  # Threshold to consider as anchor
    
    for pair_index, (operation, ocr_idx, ref_idx, cost) in enumerate(alignment_path, 1):
        
        if operation == 'SUBSTITUTE':
            # Both OCR and reference word exist
            word, word_detail = ocr_words_with_coords[ocr_idx]
            reference_word = reference_words[ref_idx]
            
            # Calculate actual similarity for status
            similarity, _, _ = word_level_similarity(word, reference_word, is_vietnamese)
            
            # Determine if this is an anchor point
            is_anchor = cost <= ANCHOR_COST_THRESHOLD
            
            # Determine status
            if is_anchor:
                status = 'MATCHED'
                alignment_type = 'ANCHOR'
            elif similarity >= threshold:
                status = 'MATCHED' 
                alignment_type = 'SUBSTITUTE'
            else:
                status = 'LOW_SIMILARITY'
                alignment_type = 'SUBSTITUTE'
            
            aligned_results.append({
                'pair_index': pair_index,
                'cluster_label': word_detail['cluster_label'],
                'ocr_word': word,
                'reference_word': reference_word,
                'similarity': similarity,
                'word_detail': word_detail,
                'status': status,
                'alignment_type': alignment_type,
                'operation': operation,
                'edit_cost': cost,
                'is_anchor': is_anchor
            })
            
        elif operation == 'DELETE':
            # Extra OCR word (no reference match)
            word, word_detail = ocr_words_with_coords[ocr_idx]
            
            aligned_results.append({
                'pair_index': pair_index,
                'cluster_label': word_detail['cluster_label'],
                'ocr_word': word,
                'reference_word': '',
                'similarity': 0.0,
                'word_detail': word_detail,
                'status': 'EXTRA_OCR',
                'alignment_type': 'DELETE',
                'operation': operation,
                'edit_cost': cost,
                'is_anchor': False
            })
            
        elif operation == 'INSERT':
            # Missing OCR word (reference exists but no OCR)
            reference_word = reference_words[ref_idx]
            
            aligned_results.append({
                'pair_index': pair_index,
                'cluster_label': f'missing_{ref_idx}',
                'ocr_word': '',
                'reference_word': reference_word,
                'similarity': 0.0,
                'word_detail': None,
                'status': 'MISSING_OCR',
                'alignment_type': 'INSERT',
                'operation': operation,
                'edit_cost': cost,
                'is_anchor': False
            })
    
    if debug:
        print(f"\nConverted {len(alignment_path)} operations to {len(aligned_results)} results")
        
        # Show operation breakdown
        ops_count = {}
        for result in aligned_results:
            op = result['operation']
            ops_count[op] = ops_count.get(op, 0) + 1
        
        print(f"Operation breakdown: {ops_count}")
    
    return aligned_results

def visualize_word_level_alignment(image, aligned_results, output_path='word_level_alignment_visualization.jpg'):
    """
    Visualize word-level alignment results with color coding for anchors and segments.
    """
    # Convert OpenCV image to PIL Image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Setup font
    font = setup_font(12)
    
    for result in aligned_results:
        pair_index = result['pair_index']
        word_detail = result.get('word_detail')
        alignment_type = result.get('alignment_type', 'UNKNOWN')
        status = result['status']
        similarity = result['similarity']
        ocr_word = result.get('ocr_word', '')
        reference_word = result.get('reference_word', '')
        
        # Skip if no word detail (missing OCR)
        if not word_detail:
            continue
        
        coordinates = word_detail['coordinates']
        
        # Color coding based on alignment type and status
        if alignment_type == 'ANCHOR':
            box_color = (0, 255, 0)  # Bright green for anchors
            text_color = (0, 200, 0)
            label_prefix = "🔗 ANCHOR"
        elif status == 'MATCHED':
            box_color = (0, 150, 255)  # Blue for good matches
            text_color = (0, 100, 200)
            label_prefix = "✅ MATCH"
        elif status == 'LOW_SIMILARITY':
            box_color = (255, 165, 0)  # Orange for low similarity
            text_color = (200, 100, 0)
            label_prefix = "⚠️ LOW"
        elif status == 'EXTRA_OCR':
            box_color = (255, 255, 0)  # Yellow for extra OCR
            text_color = (200, 200, 0)
            label_prefix = "➕ EXTRA"
        else:
            box_color = (128, 128, 128)  # Gray for others
            text_color = (100, 100, 100)
            label_prefix = "❓"
        
        # Convert coordinates to box format if needed
        if len(coordinates) == 4 and not isinstance(coordinates[0], (int, float)):
            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]
            box = np.array([[min(x_coords), min(y_coords)], [max(x_coords), min(y_coords)], 
                           [max(x_coords), max(y_coords)], [min(x_coords), max(y_coords)]])
        else:
            x_min, y_min, x_max, y_max = coordinates
            box = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        
        # Draw bounding box with thicker line for anchors
        line_width = 3 if alignment_type == 'ANCHOR' else 2
        points = [(point[0], point[1]) for point in box]
        draw.line(points + [points[0]], fill=box_color, width=line_width)
        
        # Create label
        if reference_word:
            label = f"{label_prefix}: '{ocr_word}' → '{reference_word}'"
            if alignment_type != 'ANCHOR':
                label += f" [{similarity:.2f}]"
        else:
            label = f"{label_prefix}: '{ocr_word}'"
        
        # Position label above the word
        label_pos = (int(box[0][0]), int(box[0][1] - 35))
        
        # Draw label background and text
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width, text_height = text_bbox[2], text_bbox[3]
            
            # White background for readability
            draw.rectangle(
                [label_pos[0], label_pos[1], label_pos[0] + text_width + 6, label_pos[1] + text_height + 4],
                fill=(255, 255, 255, 220)
            )
            
            # Draw text
            draw.text((label_pos[0] + 3, label_pos[1]), label, fill=text_color, font=font)
            
        except Exception as e:
            # Fallback for text drawing issues
            print(f"Warning: Could not draw label for word {pair_index}: {e}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save visualization
    pil_image.save(output_path)
    print(f"Word-level alignment visualization saved to {output_path}")

def save_word_level_alignment_results(aligned_results, output_file='word_level_alignment_results.txt'):
    """
    Save word-level alignment results to a text file.
    
    Parameters:
        aligned_results: List of word-level alignment results
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VIETNAMESE WORD-LEVEL ALIGNMENT RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            anchor_count = sum(1 for result in aligned_results if result.get('alignment_type') == 'ANCHOR')
            matched_count = sum(1 for result in aligned_results if result['status'] == 'MATCHED')
            
            f.write(f"SUMMARY:\n")
            f.write(f"  Total words: {len(aligned_results)}\n")
            f.write(f"  Anchor points: {anchor_count}\n")
            f.write(f"  High similarity matches: {matched_count}\n")
            f.write(f"  Success rate: {matched_count/len(aligned_results)*100:.1f}%\n\n")
            
            f.write("WORD-BY-WORD ALIGNMENT:\n")
            f.write("-" * 50 + "\n\n")
            
            current_cluster = None
            for result in aligned_results:
                word_index = result['pair_index']
                cluster_label = result['cluster_label']
                ocr_word = result.get('ocr_word', '')
                reference_word = result.get('reference_word', '')
                similarity = result['similarity']
                word_detail = result.get('word_detail')
                status = result['status']
                alignment_type = result.get('alignment_type', 'UNKNOWN')
                
                # Show cluster changes
                if current_cluster != cluster_label:
                    f.write(f"\n--- Cluster {cluster_label} ---\n")
                    current_cluster = cluster_label
                
                f.write(f"\nWord #{word_index} ({alignment_type}):\n")
                
                if status == 'MATCHED':
                    if alignment_type == 'ANCHOR':
                        f.write(f"  🔗 ANCHOR: '{ocr_word}' ↔ '{reference_word}' [exact match]\n")
                    else:
                        f.write(f"  ✓ MATCHED: '{ocr_word}' ↔ '{reference_word}' [similarity: {similarity:.3f}]\n")
                elif status == 'LOW_SIMILARITY':
                    f.write(f"  ⚠️ LOW SIMILARITY: '{ocr_word}' ↔ '{reference_word}' [similarity: {similarity:.3f}]\n")
                elif status == 'EXTRA_OCR':
                    f.write(f"  ➕ EXTRA OCR: '{ocr_word}' [no reference match]\n")
                elif status == 'MISSING_OCR':
                    f.write(f"  ❌ MISSING OCR: expected '{reference_word}'\n")
                
                if word_detail:
                    coords = word_detail['coordinates']
                    confidence = word_detail['confidence']
                    f.write(f"    Coordinates: {coords}\n")
                    f.write(f"    Confidence: {confidence:.3f}\n")
                
                # Show correction if applied
                if ocr_word and reference_word and ocr_word != reference_word:
                    f.write(f"    *** CORRECTION: '{ocr_word}' → '{reference_word}' ***\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Word-level alignment results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving word-level alignment results: {e}")

if __name__ == "__main__":
    main()