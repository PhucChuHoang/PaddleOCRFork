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
DET_MODEL_DIR = 'D:/University/Thesis/PaddleOCRFork/exported_det'
REC_MODEL_DIR = 'D:/University/Thesis/PaddleOCRFork/exported_rec'
REC_CHAR_DICT_PATH = 'D:/University/Thesis/PaddleOCRFork/ppocr/utils/dict/casia_hwdb_dict.txt'
NOM_DICT_PATH = 'D:/University/Thesis/PaddleOCRFork/new_dict.txt'

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
    images_folder = 'fix_img'  # Default images folder
    texts_folder = 'fix_txt'  # Default text files folder
    output_folder = 'fix_aligned'  # Default output folder
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

def save_sentence_alignment_results(aligned_results, output_file='sentence_alignment_results.txt'):
    """
    Save sentence-level alignment results to a text file.
    
    Parameters:
        aligned_results: List of sentence alignment results
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VIETNAMESE SENTENCE-LEVEL ALIGNMENT RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            matched_count = sum(1 for result in aligned_results if result['status'] == 'MATCHED')
            
            for result in aligned_results:
                cluster_label = result['cluster_label']
                ocr_sentence = result['ocr_sentence']
                reference_sentence = result['reference_sentence']
                similarity = result['similarity']
                word_details = result['word_details']
                status = result['status']
                
                f.write(f"=== Line {cluster_label} ===\n")
                f.write(f"OCR Sentence: {ocr_sentence}\n")
                
                if status == 'MATCHED':
                    f.write(f"✓ MATCHED Reference: {reference_sentence}\n")
                    f.write(f"✓ Similarity Score: {similarity:.3f}\n")
                else:
                    f.write(f"✗ NO MATCH (best similarity: {similarity:.3f})\n")
                
                f.write(f"\nWord breakdown ({len(word_details)} words):\n")
                for i, word_detail in enumerate(word_details):
                    coords = word_detail['coordinates']
                    viet_text = word_detail['vietnamese_text']
                    confidence = word_detail['confidence']
                    f.write(f"  {i+1:2d}: '{viet_text}' (conf: {confidence:.2f}, coords: {coords})\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"SUMMARY:\n")
            f.write(f"  Total lines: {len(aligned_results)}\n")
            f.write(f"  Successfully matched: {matched_count}/{len(aligned_results)}\n")
            f.write(f"  Match Rate: {matched_count/len(aligned_results)*100:.1f}%\n")
        
        print(f"Sentence alignment results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving sentence alignment results: {e}")

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

def process_sequential_sentence_alignment(img_path, reference_texts, threshold=0.5, is_vertical=True, visualize=True, debug=False, use_anchors=True):
    """
    Main function for sentence-level alignment with optional anchor-based improvement.
    
    Parameters:
        img_path: Path to input image
        reference_texts: List of reference sentence strings (in order)
        threshold: Similarity threshold for reporting quality
        is_vertical: Text layout orientation  
        visualize: Whether to create visualizations
        debug: Whether to show debug information
        use_anchors: Whether to use anchor-based alignment (recommended)
        
    Returns:
        tuple: (original_ocr_results, clustered_results, aligned_results)
    """
    # Perform OCR and clustering first
    print("Processing image with OCR and clustering...")
    original_result, clustered_result = process_image(img_path, is_vertical, visualize=False)
    
    print(f"\nFound {len(clustered_result)} {'columns' if is_vertical else 'rows'} (OCR lines)")
    print(f"Have {len(reference_texts)} reference sentences")
    
    if use_anchors:
        print("Using anchor-based alignment for improved accuracy...")
        
        # Find anchor points
        anchor_points = find_anchor_points(
            clustered_result, 
            reference_texts, 
            min_similarity=0.7,  # Lower threshold for anchor detection
            is_vietnamese=True
        )
        
        # Perform anchor-guided alignment
        aligned_results = anchor_guided_alignment(
            clustered_result, 
            reference_texts, 
            anchor_points,
            threshold=threshold, 
            is_vietnamese=True,
            debug=debug
        )
    else:
        print("Using sequential 1-to-1 alignment...")
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
    Generate training data by replacing OCR text with ground truth labels from alignment.
    
    Parameters:
        aligned_results: List of sequential alignment results
        output_format: Format for training data ('paddleocr', 'json', 'txt')
        
    Returns:
        list: Training data entries with coordinates and corrected labels
    """
    training_data = []
    
    for result in aligned_results:
        pair_index = result['pair_index']
        cluster_label = result['cluster_label']
        ocr_sentence = result['ocr_sentence']
        reference_sentence = result['reference_sentence']
        similarity = result['similarity']
        word_details = result['word_details']
        status = result['status']
        
        # Only process pairs that have both OCR and reference text
        if status in ['MATCHED', 'LOW_SIMILARITY'] and reference_sentence and word_details:
            # Normalize both texts for word alignment
            ocr_words = normalize_vietnamese_text(ocr_sentence).split()
            ref_words = normalize_vietnamese_text(reference_sentence).split()
            
            # Create word-to-word mapping
            word_mapping = align_words_for_training(ocr_words, ref_words, word_details)
            
            # Generate training entries for each word
            for word_detail, corrected_text in word_mapping:
                if word_detail and corrected_text:
                    coordinates = word_detail['coordinates']
                    original_text = word_detail['vietnamese_text']
                    confidence = word_detail['confidence']
                    
                    training_entry = {
                        'coordinates': coordinates,
                        'original_ocr': original_text,
                        'corrected_label': corrected_text,
                        'confidence': confidence,
                        'line_index': pair_index,
                        'cluster_label': cluster_label,
                        'similarity_score': similarity,
                        'status': status
                    }
                    
                    training_data.append(training_entry)
        
        elif status == 'EXTRA_OCR' and word_details:
            # For extra OCR lines, keep original OCR as labels (no ground truth available)
            for word_detail in word_details:
                coordinates = word_detail['coordinates']
                original_text = word_detail['vietnamese_text']
                confidence = word_detail['confidence']
                
                training_entry = {
                    'coordinates': coordinates,
                    'original_ocr': original_text,
                    'corrected_label': original_text,  # Keep original since no ground truth
                    'confidence': confidence,
                    'line_index': pair_index,
                    'cluster_label': cluster_label,
                    'similarity_score': 0.0,
                    'status': status
                }
                
                training_data.append(training_entry)
    
    return training_data

def align_words_for_training(ocr_words, ref_words, word_details):
    """
    Align OCR words with reference words for training data generation.
    Uses simple sequential alignment with best effort matching.
    
    Parameters:
        ocr_words: List of normalized OCR words
        ref_words: List of normalized reference words
        word_details: List of word detail dictionaries with coordinates
        
    Returns:
        list: List of (word_detail, corrected_text) tuples
    """
    word_mapping = []
    
    # Simple approach: try to align words sequentially
    min_len = min(len(ocr_words), len(ref_words), len(word_details))
    
    # Align the overlapping portion
    for i in range(min_len):
        word_detail = word_details[i]
        corrected_text = ref_words[i]
        word_mapping.append((word_detail, corrected_text))
    
    # Handle remaining OCR words (if OCR has more words than reference)
    if len(word_details) > min_len:
        for i in range(min_len, len(word_details)):
            word_detail = word_details[i]
            # Use original OCR text if no corresponding reference word
            original_normalized = normalize_vietnamese_text(word_detail['vietnamese_text'])
            word_mapping.append((word_detail, original_normalized))
    
    return word_mapping

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
        # Group by line for better organization
        current_line = None
        line_entries = []
        
        for entry in training_data:
            line_index = entry['line_index']
            
            if current_line != line_index:
                # Write previous line if exists
                if line_entries:
                    write_paddleocr_line(f, current_line, line_entries, actual_image_path)
                
                # Start new line
                current_line = line_index
                line_entries = [entry]
            else:
                line_entries.append(entry)
        
        # Write last line
        if line_entries:
            write_paddleocr_line(f, current_line, line_entries, actual_image_path)

def write_paddleocr_line(f, line_index, entries, actual_image_path=None):
    """Write a single line in PaddleOCR format"""
    # Use actual image path if provided, otherwise use dummy path
    if actual_image_path:
        image_path = os.path.basename(actual_image_path)
    else:
        image_path = f"line_{line_index}.jpg"
    
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
        f.write("TRAINING DATA FOR SEMI-SUPERVISED LEARNING\n")
        f.write("=" * 60 + "\n\n")
        
        current_line = None
        for entry in training_data:
            line_index = entry['line_index']
            
            if current_line != line_index:
                f.write(f"\n=== Line {line_index} ===\n")
                current_line = line_index
            
            coords = entry['coordinates']
            original = entry['original_ocr']
            corrected = entry['corrected_label']
            confidence = entry['confidence']
            status = entry['status']
            
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
            save_sentence_alignment_results(aligned_results, 
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

def find_anchor_points(clustered_result, reference_texts, min_similarity=0.8, is_vietnamese=True):
    """
    Find high-confidence anchor points between OCR lines and reference texts.
    These anchors will guide the overall alignment process.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        reference_texts: List of reference sentences
        min_similarity: Minimum similarity threshold for anchor points
        is_vietnamese: Whether to use Vietnamese normalization
        
    Returns:
        list: List of anchor points as (ocr_index, ref_index, similarity) tuples
    """
    nom_dict = load_nom_dictionary()
    sorted_clusters = sorted(clustered_result.keys())
    anchor_points = []
    
    print(f"Finding anchor points with similarity >= {min_similarity}...")
    
    # Compare each OCR line with each reference text
    for ocr_idx, cluster_label in enumerate(sorted_clusters):
        cluster_lines = clustered_result[cluster_label]
        ocr_sentence, _, _ = combine_cluster_to_sentence(cluster_lines, nom_dict)
        
        if not ocr_sentence.strip():
            continue
            
        best_similarity = 0.0
        best_ref_idx = -1
        
        for ref_idx, reference_text in enumerate(reference_texts):
            if not reference_text.strip():
                continue
                
            similarity, _, _ = word_level_similarity(ocr_sentence, reference_text, is_vietnamese)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_ref_idx = ref_idx
        
        # Add as anchor point if similarity is high enough
        if best_similarity >= min_similarity and best_ref_idx >= 0:
            anchor_points.append((ocr_idx, best_ref_idx, best_similarity))
    
    # Sort anchor points by OCR index to maintain order
    anchor_points.sort(key=lambda x: x[0])
    
    print(f"Found {len(anchor_points)} anchor points:")
    for ocr_idx, ref_idx, sim in anchor_points:
        print(f"  OCR line {ocr_idx+1} ↔ Reference line {ref_idx+1} (similarity: {sim:.3f})")
    
    return anchor_points

def anchor_guided_alignment(clustered_result, reference_texts, anchor_points, threshold=0.4, is_vietnamese=True, debug=False):
    """
    Perform alignment guided by anchor points. This creates segments between anchors
    and aligns each segment independently.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        reference_texts: List of reference sentences
        anchor_points: List of (ocr_index, ref_index, similarity) anchor tuples
        threshold: Similarity threshold for reporting quality
        is_vietnamese: Whether to use Vietnamese normalization
        debug: Whether to show debug information
        
    Returns:
        list: Aligned results with improved accuracy
    """
    nom_dict = load_nom_dictionary()
    sorted_clusters = sorted(clustered_result.keys())
    aligned_results = []
    
    if debug:
        print(f"\n=== ANCHOR-GUIDED ALIGNMENT ===")
        print(f"OCR lines: {len(sorted_clusters)}")
        print(f"Reference lines: {len(reference_texts)}")
        print(f"Anchor points: {len(anchor_points)}")
    
    # If no anchor points, fall back to sequential alignment
    if not anchor_points:
        print("No anchor points found, falling back to sequential alignment...")
        return find_sequential_sentence_alignment(clustered_result, reference_texts, threshold, is_vietnamese, debug)
    
    # Create segments between anchor points
    segments = []
    prev_ocr_idx = 0
    prev_ref_idx = 0
    
    for ocr_idx, ref_idx, similarity in anchor_points:
        # Add segment before this anchor (if any)
        if ocr_idx > prev_ocr_idx or ref_idx > prev_ref_idx:
            segments.append({
                'ocr_start': prev_ocr_idx,
                'ocr_end': ocr_idx,
                'ref_start': prev_ref_idx,
                'ref_end': ref_idx,
                'type': 'segment'
            })
        
        # Add the anchor point itself
        segments.append({
            'ocr_start': ocr_idx,
            'ocr_end': ocr_idx + 1,
            'ref_start': ref_idx,
            'ref_end': ref_idx + 1,
            'type': 'anchor',
            'similarity': similarity
        })
        
        prev_ocr_idx = ocr_idx + 1
        prev_ref_idx = ref_idx + 1
    
    # Add final segment after last anchor (if any)
    if prev_ocr_idx < len(sorted_clusters) or prev_ref_idx < len(reference_texts):
        segments.append({
            'ocr_start': prev_ocr_idx,
            'ocr_end': len(sorted_clusters),
            'ref_start': prev_ref_idx,
            'ref_end': len(reference_texts),
            'type': 'segment'
        })
    
    if debug:
        print(f"Created {len(segments)} segments:")
        for i, seg in enumerate(segments):
            print(f"  Segment {i+1}: OCR[{seg['ocr_start']}:{seg['ocr_end']}] ↔ Ref[{seg['ref_start']}:{seg['ref_end']}] ({seg['type']})")
    
    # Align each segment
    pair_index = 1
    for segment in segments:
        if segment['type'] == 'anchor':
            # Direct alignment for anchor points
            ocr_idx = segment['ocr_start']
            ref_idx = segment['ref_start']
            
            cluster_label = sorted_clusters[ocr_idx]
            cluster_lines = clustered_result[cluster_label]
            ocr_sentence, word_details, coordinates_list = combine_cluster_to_sentence(cluster_lines, nom_dict)
            reference_sentence = reference_texts[ref_idx]
            similarity = segment['similarity']
            
            aligned_results.append({
                'pair_index': pair_index,
                'cluster_label': cluster_label,
                'ocr_sentence': ocr_sentence,
                'reference_sentence': reference_sentence,
                'similarity': similarity,
                'word_details': word_details,
                'coordinates_list': coordinates_list,
                'status': 'MATCHED',
                'alignment_type': 'ANCHOR'
            })
            pair_index += 1
            
        else:
            # Align segment using optimal strategy
            segment_results = align_segment(
                clustered_result, reference_texts, sorted_clusters, segment, 
                nom_dict, threshold, is_vietnamese, pair_index, debug
            )
            aligned_results.extend(segment_results)
            pair_index += len(segment_results)
    
    if debug:
        matched_count = sum(1 for r in aligned_results if r['status'] == 'MATCHED')
        anchor_count = sum(1 for r in aligned_results if r.get('alignment_type') == 'ANCHOR')
        print(f"\n=== ANCHOR-GUIDED ALIGNMENT SUMMARY ===")
        print(f"Total pairs: {len(aligned_results)}")
        print(f"Anchor alignments: {anchor_count}")
        print(f"High similarity matches: {matched_count}")
        print(f"Success rate: {matched_count/len(aligned_results)*100:.1f}%")
    
    return aligned_results

def align_segment(clustered_result, reference_texts, sorted_clusters, segment, nom_dict, threshold, is_vietnamese, start_pair_index, debug=False):
    """
    Align a segment between anchor points using the best strategy based on segment size.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        reference_texts: List of reference sentences
        sorted_clusters: Sorted cluster labels
        segment: Segment dictionary with start/end indices
        nom_dict: Vietnamese dictionary
        threshold: Similarity threshold
        is_vietnamese: Whether to use Vietnamese normalization
        start_pair_index: Starting pair index for numbering
        debug: Whether to show debug information
        
    Returns:
        list: Aligned results for this segment
    """
    ocr_start, ocr_end = segment['ocr_start'], segment['ocr_end']
    ref_start, ref_end = segment['ref_start'], segment['ref_end']
    
    ocr_count = ocr_end - ocr_start
    ref_count = ref_end - ref_start
    
    if debug:
        print(f"Aligning segment: {ocr_count} OCR lines vs {ref_count} reference lines")
    
    segment_results = []
    
    if ocr_count == 0 and ref_count == 0:
        # Empty segment, nothing to align
        return segment_results
    
    elif ocr_count == 0:
        # Missing OCR lines
        for i in range(ref_count):
            ref_idx = ref_start + i
            segment_results.append({
                'pair_index': start_pair_index + i,
                'cluster_label': f'missing_{ref_idx}',
                'ocr_sentence': '',
                'reference_sentence': reference_texts[ref_idx],
                'similarity': 0.0,
                'word_details': [],
                'coordinates_list': [],
                'status': 'MISSING_OCR',
                'alignment_type': 'SEGMENT'
            })
    
    elif ref_count == 0:
        # Extra OCR lines
        for i in range(ocr_count):
            ocr_idx = ocr_start + i
            cluster_label = sorted_clusters[ocr_idx]
            cluster_lines = clustered_result[cluster_label]
            ocr_sentence, word_details, coordinates_list = combine_cluster_to_sentence(cluster_lines, nom_dict)
            
            segment_results.append({
                'pair_index': start_pair_index + i,
                'cluster_label': cluster_label,
                'ocr_sentence': ocr_sentence,
                'reference_sentence': '',
                'similarity': 0.0,
                'word_details': word_details,
                'coordinates_list': coordinates_list,
                'status': 'EXTRA_OCR',
                'alignment_type': 'SEGMENT'
            })
    
    elif ocr_count == ref_count:
        # Equal counts - use sequential alignment
        for i in range(ocr_count):
            ocr_idx = ocr_start + i
            ref_idx = ref_start + i
            
            cluster_label = sorted_clusters[ocr_idx]
            cluster_lines = clustered_result[cluster_label]
            ocr_sentence, word_details, coordinates_list = combine_cluster_to_sentence(cluster_lines, nom_dict)
            reference_sentence = reference_texts[ref_idx]
            
            similarity, _, _ = word_level_similarity(ocr_sentence, reference_sentence, is_vietnamese)
            status = 'MATCHED' if similarity >= threshold else 'LOW_SIMILARITY'
            
            segment_results.append({
                'pair_index': start_pair_index + i,
                'cluster_label': cluster_label,
                'ocr_sentence': ocr_sentence,
                'reference_sentence': reference_sentence,
                'similarity': similarity,
                'word_details': word_details,
                'coordinates_list': coordinates_list,
                'status': status,
                'alignment_type': 'SEGMENT'
            })
    
    else:
        # Different counts - use optimal alignment within segment
        segment_results = optimal_segment_alignment(
            clustered_result, reference_texts, sorted_clusters, segment, 
            nom_dict, threshold, is_vietnamese, start_pair_index, debug
        )
    
    return segment_results

def optimal_segment_alignment(clustered_result, reference_texts, sorted_clusters, segment, nom_dict, threshold, is_vietnamese, start_pair_index, debug=False):
    """
    Perform optimal alignment within a segment using dynamic programming approach.
    This handles cases where OCR and reference counts don't match.
    
    Parameters:
        clustered_result: Dictionary of clustered OCR results
        reference_texts: List of reference sentences
        sorted_clusters: Sorted cluster labels
        segment: Segment dictionary with start/end indices
        nom_dict: Vietnamese dictionary
        threshold: Similarity threshold
        is_vietnamese: Whether to use Vietnamese normalization
        start_pair_index: Starting pair index for numbering
        debug: Whether to show debug information
        
    Returns:
        list: Optimally aligned results for this segment
    """
    ocr_start, ocr_end = segment['ocr_start'], segment['ocr_end']
    ref_start, ref_end = segment['ref_start'], segment['ref_end']
    
    ocr_indices = list(range(ocr_start, ocr_end))
    ref_indices = list(range(ref_start, ref_end))
    
    if debug:
        print(f"Optimal alignment: {len(ocr_indices)} OCR vs {len(ref_indices)} ref")
    
    # Get OCR sentences
    ocr_sentences = []
    ocr_details = []
    for ocr_idx in ocr_indices:
        cluster_label = sorted_clusters[ocr_idx]
        cluster_lines = clustered_result[cluster_label]
        ocr_sentence, word_details, coordinates_list = combine_cluster_to_sentence(cluster_lines, nom_dict)
        ocr_sentences.append(ocr_sentence)
        ocr_details.append((cluster_label, word_details, coordinates_list))
    
    # Calculate similarity matrix
    similarity_matrix = []
    for ocr_sentence in ocr_sentences:
        row = []
        for ref_idx in ref_indices:
            reference_sentence = reference_texts[ref_idx]
            similarity, _, _ = word_level_similarity(ocr_sentence, reference_sentence, is_vietnamese)
            row.append(similarity)
        similarity_matrix.append(row)
    
    # Find optimal alignment using greedy approach (can be improved with DP)
    alignments = greedy_alignment(similarity_matrix, threshold)
    
    # Create results
    segment_results = []
    pair_index = start_pair_index
    
    for ocr_local_idx, ref_local_idx, similarity in alignments:
        if ocr_local_idx >= 0 and ref_local_idx >= 0:
            # Both OCR and reference exist
            ocr_idx = ocr_indices[ocr_local_idx]
            ref_idx = ref_indices[ref_local_idx]
            
            cluster_label, word_details, coordinates_list = ocr_details[ocr_local_idx]
            ocr_sentence = ocr_sentences[ocr_local_idx]
            reference_sentence = reference_texts[ref_idx]
            status = 'MATCHED' if similarity >= threshold else 'LOW_SIMILARITY'
            
        elif ocr_local_idx >= 0:
            # Extra OCR
            ocr_idx = ocr_indices[ocr_local_idx]
            cluster_label, word_details, coordinates_list = ocr_details[ocr_local_idx]
            ocr_sentence = ocr_sentences[ocr_local_idx]
            reference_sentence = ''
            similarity = 0.0
            status = 'EXTRA_OCR'
            
        else:
            # Missing OCR
            ref_idx = ref_indices[ref_local_idx]
            cluster_label = f'missing_{ref_idx}'
            word_details = []
            coordinates_list = []
            ocr_sentence = ''
            reference_sentence = reference_texts[ref_idx]
            similarity = 0.0
            status = 'MISSING_OCR'
        
        segment_results.append({
            'pair_index': pair_index,
            'cluster_label': cluster_label,
            'ocr_sentence': ocr_sentence,
            'reference_sentence': reference_sentence,
            'similarity': similarity,
            'word_details': word_details,
            'coordinates_list': coordinates_list,
            'status': status,
            'alignment_type': 'OPTIMAL'
        })
        pair_index += 1
    
    return segment_results

def greedy_alignment(similarity_matrix, threshold=0.3):
    """
    Perform greedy alignment based on similarity matrix.
    
    Parameters:
        similarity_matrix: 2D list of similarities [ocr_count][ref_count]
        threshold: Minimum similarity for considering a match
        
    Returns:
        list: List of (ocr_idx, ref_idx, similarity) tuples, with -1 for unmatched
    """
    if not similarity_matrix or not similarity_matrix[0]:
        return []
    
    ocr_count = len(similarity_matrix)
    ref_count = len(similarity_matrix[0])
    
    # Find all potential matches above threshold
    potential_matches = []
    for i in range(ocr_count):
        for j in range(ref_count):
            if similarity_matrix[i][j] >= threshold:
                potential_matches.append((i, j, similarity_matrix[i][j]))
    
    # Sort by similarity (descending)
    potential_matches.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy selection
    used_ocr = set()
    used_ref = set()
    alignments = []
    
    for ocr_idx, ref_idx, similarity in potential_matches:
        if ocr_idx not in used_ocr and ref_idx not in used_ref:
            alignments.append((ocr_idx, ref_idx, similarity))
            used_ocr.add(ocr_idx)
            used_ref.add(ref_idx)
    
    # Add unmatched OCR lines
    for i in range(ocr_count):
        if i not in used_ocr:
            alignments.append((i, -1, 0.0))
    
    # Add unmatched reference lines
    for j in range(ref_count):
        if j not in used_ref:
            alignments.append((-1, j, 0.0))
    
    # Sort by OCR index for consistent ordering
    alignments.sort(key=lambda x: (x[0] if x[0] >= 0 else float('inf'), x[1] if x[1] >= 0 else float('inf')))
    
    return alignments

if __name__ == "__main__":
    main()