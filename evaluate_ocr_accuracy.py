#!/usr/bin/env python3
"""
OCR Accuracy Evaluation Script
Based on word-level anchor-based alignment from align.py

This script evaluates OCR model accuracy using various metrics:
- Word-level accuracy (exact matches)
- Character-level accuracy 
- Edit distance metrics
- Anchor point accuracy
- Confidence-based analysis
- Per-class character accuracy

Usage:
    python evaluate_ocr_accuracy.py --images_folder path/to/images --texts_folder path/to/ground_truth --output_folder path/to/results
"""

import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# Import functions from align.py
from align import (
    process_sequential_sentence_alignment,
    find_matching_files,
    load_ground_truth_file,
    word_level_similarity,
    levenshtein_distance,
    normalize_text,
    load_nom_dictionary,
    get_vietnamese_text
)

class OCRAccuracyEvaluator:
    """Comprehensive OCR accuracy evaluation system."""
    
    def __init__(self, is_vietnamese=True, min_anchor_similarity=0.9, similarity_threshold=0.6):
        """
        Initialize the evaluator.
        
        Parameters:
            is_vietnamese: Whether to use Vietnamese-specific normalization
            min_anchor_similarity: Minimum similarity for anchor points
            similarity_threshold: Threshold for considering matches as accurate
        """
        self.is_vietnamese = is_vietnamese
        self.min_anchor_similarity = min_anchor_similarity
        self.similarity_threshold = similarity_threshold
        self.nom_dict = load_nom_dictionary() if is_vietnamese else None
        
        # Initialize metrics storage
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self.word_metrics = {
            'total_words': 0,
            'exact_matches': 0,
            'anchor_matches': 0,
            'high_similarity_matches': 0,
            'low_similarity_matches': 0,
            'extra_ocr_words': 0,
            'missing_ocr_words': 0,
            'total_similarity_score': 0.0
        }
        
        self.character_metrics = {
            'total_characters': 0,
            'correct_characters': 0,
            'total_edit_distance': 0
        }
        
        self.confidence_metrics = {
            'high_confidence_correct': 0,
            'high_confidence_total': 0,
            'low_confidence_correct': 0,
            'low_confidence_total': 0,
            'confidence_scores': [],
            'accuracy_scores': []
        }
        
        self.per_character_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.alignment_costs = []
        self.file_results = []
    
    def evaluate_single_file(self, image_path, text_path, debug=False):
        """
        Evaluate OCR accuracy for a single image-text pair.
        
        Parameters:
            image_path: Path to the image file
            text_path: Path to the ground truth text file
            debug: Whether to show debug information
            
        Returns:
            dict: Evaluation results for this file
        """
        try:
            # Load ground truth
            reference_texts = load_ground_truth_file(text_path)
            if not reference_texts:
                print(f"Warning: No ground truth found for {text_path}")
                return None
            
            base_name = Path(image_path).stem
            
            if debug:
                print(f"\n{'='*60}")
                print(f"EVALUATING: {base_name}")
                print(f"{'='*60}")
            
            # Perform OCR and alignment
            original_result, clustered_result, aligned_results = process_sequential_sentence_alignment(
                img_path=image_path,
                reference_texts=reference_texts,
                threshold=self.similarity_threshold,
                is_vertical=True,
                visualize=False,
                debug=debug,
                use_anchors=True
            )
            
            # Calculate metrics for this file
            file_metrics = self.calculate_file_metrics(aligned_results, base_name, debug)
            
            # Update accumulated metrics
            self.update_accumulated_metrics(aligned_results)
            
            self.file_results.append(file_metrics)
            
            return file_metrics
            
        except Exception as e:
            print(f"Error evaluating {image_path}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None
    
    def calculate_file_metrics(self, aligned_results, file_name, debug=False):
        """Calculate accuracy metrics for a single file."""
        
        # Initialize file-level counters
        file_word_metrics = {
            'total_words': 0,
            'exact_matches': 0,
            'anchor_matches': 0,
            'high_similarity_matches': 0,
            'low_similarity_matches': 0,
            'extra_ocr_words': 0,
            'missing_ocr_words': 0,
            'total_similarity_score': 0.0
        }
        
        file_char_metrics = {
            'total_characters': 0,
            'correct_characters': 0,
            'total_edit_distance': 0
        }
        
        confidence_data = []
        
        # Process each aligned word
        for result in aligned_results:
            file_word_metrics['total_words'] += 1
            
            ocr_word = result.get('ocr_word', '')
            reference_word = result.get('reference_word', '')
            similarity = result['similarity']
            status = result['status']
            alignment_type = result.get('alignment_type', 'UNKNOWN')
            word_detail = result.get('word_detail')
            
            # Word-level metrics
            file_word_metrics['total_similarity_score'] += similarity
            
            if status == 'MATCHED':
                if alignment_type == 'ANCHOR':
                    file_word_metrics['anchor_matches'] += 1
                    file_word_metrics['exact_matches'] += 1
                elif similarity >= self.similarity_threshold:
                    file_word_metrics['high_similarity_matches'] += 1
                else:
                    file_word_metrics['low_similarity_matches'] += 1
            elif status == 'LOW_SIMILARITY':
                file_word_metrics['low_similarity_matches'] += 1
            elif status == 'EXTRA_OCR':
                file_word_metrics['extra_ocr_words'] += 1
            elif status == 'MISSING_OCR':
                file_word_metrics['missing_ocr_words'] += 1
            
            # Character-level metrics
            if ocr_word and reference_word:
                char_edit_dist = levenshtein_distance(ocr_word, reference_word)
                max_len = max(len(ocr_word), len(reference_word))
                
                file_char_metrics['total_edit_distance'] += char_edit_dist
                file_char_metrics['total_characters'] += max_len
                file_char_metrics['correct_characters'] += max_len - char_edit_dist
            
            # Confidence analysis
            if word_detail:
                confidence = word_detail['confidence']
                is_correct = (alignment_type == 'ANCHOR' or 
                            (status == 'MATCHED' and similarity >= self.similarity_threshold))
                confidence_data.append((confidence, is_correct))
        
        # Calculate file-level accuracy rates
        total_processable = (file_word_metrics['exact_matches'] + 
                           file_word_metrics['high_similarity_matches'] + 
                           file_word_metrics['low_similarity_matches'] + 
                           file_word_metrics['extra_ocr_words'])
        
        file_accuracy_metrics = {
            'word_accuracy': (file_word_metrics['exact_matches'] + file_word_metrics['high_similarity_matches']) / max(total_processable, 1) * 100,
            'anchor_accuracy': file_word_metrics['anchor_matches'] / max(total_processable, 1) * 100,
            'character_accuracy': file_char_metrics['correct_characters'] / max(file_char_metrics['total_characters'], 1) * 100,
            'average_similarity': file_word_metrics['total_similarity_score'] / max(file_word_metrics['total_words'], 1),
            'edit_distance_ratio': file_char_metrics['total_edit_distance'] / max(file_char_metrics['total_characters'], 1)
        }
        
        if debug:
            print(f"\nFile Metrics for {file_name}:")
            print(f"  Word Accuracy: {file_accuracy_metrics['word_accuracy']:.2f}%")
            print(f"  Anchor Accuracy: {file_accuracy_metrics['anchor_accuracy']:.2f}%")
            print(f"  Character Accuracy: {file_accuracy_metrics['character_accuracy']:.2f}%")
            print(f"  Average Similarity: {file_accuracy_metrics['average_similarity']:.3f}")
            print(f"  Total Words: {file_word_metrics['total_words']}")
        
        return {
            'file_name': file_name,
            'word_metrics': file_word_metrics,
            'character_metrics': file_char_metrics,
            'accuracy_metrics': file_accuracy_metrics,
            'confidence_data': confidence_data
        }
    
    def update_accumulated_metrics(self, aligned_results):
        """Update accumulated metrics across all files."""
        
        for result in aligned_results:
            self.word_metrics['total_words'] += 1
            
            ocr_word = result.get('ocr_word', '')
            reference_word = result.get('reference_word', '')
            similarity = result['similarity']
            status = result['status']
            alignment_type = result.get('alignment_type', 'UNKNOWN')
            word_detail = result.get('word_detail')
            edit_cost = result.get('edit_cost', 0)
            
            # Word-level metrics
            self.word_metrics['total_similarity_score'] += similarity
            self.alignment_costs.append(edit_cost)
            
            if status == 'MATCHED':
                if alignment_type == 'ANCHOR':
                    self.word_metrics['anchor_matches'] += 1
                    self.word_metrics['exact_matches'] += 1
                elif similarity >= self.similarity_threshold:
                    self.word_metrics['high_similarity_matches'] += 1
                else:
                    self.word_metrics['low_similarity_matches'] += 1
            elif status == 'LOW_SIMILARITY':
                self.word_metrics['low_similarity_matches'] += 1
            elif status == 'EXTRA_OCR':
                self.word_metrics['extra_ocr_words'] += 1
            elif status == 'MISSING_OCR':
                self.word_metrics['missing_ocr_words'] += 1
            
            # Character-level metrics
            if ocr_word and reference_word:
                char_edit_dist = levenshtein_distance(ocr_word, reference_word)
                max_len = max(len(ocr_word), len(reference_word))
                
                self.character_metrics['total_edit_distance'] += char_edit_dist
                self.character_metrics['total_characters'] += max_len
                self.character_metrics['correct_characters'] += max_len - char_edit_dist
            
            # Confidence analysis
            if word_detail:
                confidence = word_detail['confidence']
                is_correct = (alignment_type == 'ANCHOR' or 
                            (status == 'MATCHED' and similarity >= self.similarity_threshold))
                
                self.confidence_metrics['confidence_scores'].append(confidence)
                self.confidence_metrics['accuracy_scores'].append(1.0 if is_correct else 0.0)
                
                # High vs low confidence analysis
                if confidence >= 0.8:  # High confidence threshold
                    self.confidence_metrics['high_confidence_total'] += 1
                    if is_correct:
                        self.confidence_metrics['high_confidence_correct'] += 1
                else:  # Low confidence
                    self.confidence_metrics['low_confidence_total'] += 1
                    if is_correct:
                        self.confidence_metrics['low_confidence_correct'] += 1
    
    def calculate_overall_accuracy(self):
        """Calculate overall accuracy metrics."""
        
        total_processable = (self.word_metrics['exact_matches'] + 
                           self.word_metrics['high_similarity_matches'] + 
                           self.word_metrics['low_similarity_matches'] + 
                           self.word_metrics['extra_ocr_words'])
        
        overall_metrics = {
            'word_accuracy': (self.word_metrics['exact_matches'] + self.word_metrics['high_similarity_matches']) / max(total_processable, 1) * 100,
            'exact_word_accuracy': self.word_metrics['exact_matches'] / max(total_processable, 1) * 100,
            'anchor_accuracy': self.word_metrics['anchor_matches'] / max(total_processable, 1) * 100,
            'character_accuracy': self.character_metrics['correct_characters'] / max(self.character_metrics['total_characters'], 1) * 100,
            'average_similarity': self.word_metrics['total_similarity_score'] / max(self.word_metrics['total_words'], 1),
            'edit_distance_ratio': self.character_metrics['total_edit_distance'] / max(self.character_metrics['total_characters'], 1),
            'average_alignment_cost': np.mean(self.alignment_costs) if self.alignment_costs else 0,
        }
        
        # Confidence-based metrics
        if self.confidence_metrics['high_confidence_total'] > 0:
            overall_metrics['high_confidence_accuracy'] = self.confidence_metrics['high_confidence_correct'] / self.confidence_metrics['high_confidence_total'] * 100
        else:
            overall_metrics['high_confidence_accuracy'] = 0
            
        if self.confidence_metrics['low_confidence_total'] > 0:
            overall_metrics['low_confidence_accuracy'] = self.confidence_metrics['low_confidence_correct'] / self.confidence_metrics['low_confidence_total'] * 100
        else:
            overall_metrics['low_confidence_accuracy'] = 0
        
        # Error rate metrics
        overall_metrics['error_rate'] = 100 - overall_metrics['word_accuracy']
        overall_metrics['substitution_rate'] = self.word_metrics['low_similarity_matches'] / max(total_processable, 1) * 100
        overall_metrics['insertion_rate'] = self.word_metrics['extra_ocr_words'] / max(total_processable, 1) * 100
        overall_metrics['deletion_rate'] = self.word_metrics['missing_ocr_words'] / max(self.word_metrics['total_words'], 1) * 100
        
        return overall_metrics
    
    def generate_evaluation_report(self, output_folder):
        """Generate comprehensive evaluation report."""
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_accuracy()
        
        # Generate text report
        self.generate_text_report(overall_metrics, output_folder)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT GENERATED")
        print(f"{'='*60}")
        print(f"Word Accuracy: {overall_metrics['word_accuracy']:.2f}%")
        print(f"Character Accuracy: {overall_metrics['character_accuracy']:.2f}%")
        print(f"Anchor Point Accuracy: {overall_metrics['anchor_accuracy']:.2f}%")
        print(f"Average Similarity: {overall_metrics['average_similarity']:.3f}")
        print(f"Total Files Processed: {len(self.file_results)}")
        print(f"Total Words Evaluated: {self.word_metrics['total_words']}")
        print(f"\nDetailed report saved in: {output_folder}")
    
    def generate_text_report(self, overall_metrics, output_folder):
        """Generate detailed text report."""
        
        report_path = os.path.join(output_folder, 'ocr_accuracy_report_old_model.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("OCR ACCURACY EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL ACCURACY METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Word Accuracy: {overall_metrics['word_accuracy']:.2f}%\n")
            f.write(f"Exact Word Accuracy: {overall_metrics['exact_word_accuracy']:.2f}%\n")
            f.write(f"Anchor Point Accuracy: {overall_metrics['anchor_accuracy']:.2f}%\n")
            f.write(f"Character Accuracy: {overall_metrics['character_accuracy']:.2f}%\n")
            f.write(f"Average Word Similarity: {overall_metrics['average_similarity']:.3f}\n")
            f.write(f"Edit Distance Ratio: {overall_metrics['edit_distance_ratio']:.3f}\n")
            f.write(f"Average Alignment Cost: {overall_metrics['average_alignment_cost']:.3f}\n\n")
            
            f.write("ERROR ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Error Rate: {overall_metrics['error_rate']:.2f}%\n")
            f.write(f"Substitution Error Rate: {overall_metrics['substitution_rate']:.2f}%\n")
            f.write(f"Insertion Error Rate: {overall_metrics['insertion_rate']:.2f}%\n")
            f.write(f"Deletion Error Rate: {overall_metrics['deletion_rate']:.2f}%\n\n")
            
            f.write("CONFIDENCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"High Confidence Accuracy: {overall_metrics['high_confidence_accuracy']:.2f}%\n")
            f.write(f"Low Confidence Accuracy: {overall_metrics['low_confidence_accuracy']:.2f}%\n")
            f.write(f"High Confidence Samples: {self.confidence_metrics['high_confidence_total']}\n")
            f.write(f"Low Confidence Samples: {self.confidence_metrics['low_confidence_total']}\n\n")
            
            f.write("WORD-LEVEL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Words Processed: {self.word_metrics['total_words']}\n")
            f.write(f"Exact Matches: {self.word_metrics['exact_matches']}\n")
            f.write(f"Anchor Matches: {self.word_metrics['anchor_matches']}\n")
            f.write(f"High Similarity Matches: {self.word_metrics['high_similarity_matches']}\n")
            f.write(f"Low Similarity Matches: {self.word_metrics['low_similarity_matches']}\n")
            f.write(f"Extra OCR Words: {self.word_metrics['extra_ocr_words']}\n")
            f.write(f"Missing OCR Words: {self.word_metrics['missing_ocr_words']}\n\n")
            
            f.write("CHARACTER-LEVEL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Characters: {self.character_metrics['total_characters']}\n")
            f.write(f"Correct Characters: {self.character_metrics['correct_characters']}\n")
            f.write(f"Total Edit Distance: {self.character_metrics['total_edit_distance']}\n\n")
            
            f.write("PER-FILE RESULTS:\n")
            f.write("-" * 30 + "\n")
            for file_result in self.file_results:
                f.write(f"\nFile: {file_result['file_name']}\n")
                f.write(f"  Word Accuracy: {file_result['accuracy_metrics']['word_accuracy']:.2f}%\n")
                f.write(f"  Character Accuracy: {file_result['accuracy_metrics']['character_accuracy']:.2f}%\n")
                f.write(f"  Average Similarity: {file_result['accuracy_metrics']['average_similarity']:.3f}\n")
                f.write(f"  Total Words: {file_result['word_metrics']['total_words']}\n")
        
        print(f"Text report saved to {report_path}")


def evaluate_batch_accuracy(images_folder, texts_folder, output_folder, 
                          min_anchor_similarity=0.9, similarity_threshold=0.6, 
                          is_vietnamese=True, debug=False):
    """
    Evaluate OCR accuracy for a batch of image-text pairs.
    
    Parameters:
        images_folder: Path to folder containing images
        texts_folder: Path to folder containing ground truth text files
        output_folder: Path to output folder for evaluation results
        min_anchor_similarity: Minimum similarity for anchor points
        similarity_threshold: Threshold for considering matches as accurate
        is_vietnamese: Whether to use Vietnamese-specific processing
        debug: Whether to show debug information
        
    Returns:
        OCRAccuracyEvaluator: Evaluator instance with results
    """
    
    # Initialize evaluator
    evaluator = OCRAccuracyEvaluator(
        is_vietnamese=is_vietnamese,
        min_anchor_similarity=min_anchor_similarity,
        similarity_threshold=similarity_threshold
    )
    
    # Find matching files
    matching_pairs = find_matching_files(images_folder, texts_folder)
    
    if not matching_pairs:
        print("No matching image-text pairs found!")
        return None
    
    print(f"\n{'='*80}")
    print(f"OCR ACCURACY EVALUATION: {len(matching_pairs)} FILES")
    print(f"{'='*80}")
    print(f"Images folder: {images_folder}")
    print(f"Ground truth folder: {texts_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Anchor similarity threshold: {min_anchor_similarity}")
    print(f"Match similarity threshold: {similarity_threshold}")
    
    # Process each file
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, (image_path, text_path, base_name) in enumerate(matching_pairs, 1):
        print(f"\n--- Evaluating {i}/{len(matching_pairs)}: {base_name} ---")
        
        file_result = evaluator.evaluate_single_file(
            image_path, 
            text_path, 
            debug=(debug and i <= 3)  # Debug first 3 files only
        )
        
        if file_result:
            successful_evaluations += 1
            word_acc = file_result['accuracy_metrics']['word_accuracy']
            char_acc = file_result['accuracy_metrics']['character_accuracy']
            print(f"✅ {base_name}: Word {word_acc:.1f}%, Char {char_acc:.1f}%")
        else:
            failed_evaluations += 1
            print(f"❌ {base_name}: Evaluation failed")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETED")
    print(f"{'='*60}")
    print(f"Successfully evaluated: {successful_evaluations}/{len(matching_pairs)}")
    print(f"Failed evaluations: {failed_evaluations}")
    
    # Generate comprehensive report
    if successful_evaluations > 0:
        evaluator.generate_evaluation_report(output_folder)
    else:
        print("No successful evaluations - cannot generate report")
    
    return evaluator


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate OCR model accuracy using word-level anchor-based alignment"
    )
    
    parser.add_argument('--images_folder', required=True,
                       help='Path to folder containing test images')
    parser.add_argument('--texts_folder', required=True,
                       help='Path to folder containing ground truth text files')
    parser.add_argument('--output_folder', default='evaluation_results',
                       help='Path to output folder for evaluation results')
    parser.add_argument('--anchor_threshold', type=float, default=0.9,
                       help='Minimum similarity for anchor points (default: 0.9)')
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                       help='Similarity threshold for accurate matches (default: 0.6)')
    parser.add_argument('--is_vietnamese', action='store_true', default=True,
                       help='Use Vietnamese-specific text processing (default: True)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for detailed output')
    
    args = parser.parse_args()
    
    # Validate input folders
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder '{args.images_folder}' does not exist!")
        return 1
    
    if not os.path.exists(args.texts_folder):
        print(f"Error: Text files folder '{args.texts_folder}' does not exist!")
        return 1
    
    # Run evaluation
    evaluator = evaluate_batch_accuracy(
        images_folder=args.images_folder,
        texts_folder=args.texts_folder,
        output_folder=args.output_folder,
        min_anchor_similarity=args.anchor_threshold,
        similarity_threshold=args.similarity_threshold,
        is_vietnamese=args.is_vietnamese,
        debug=args.debug
    )
    
    if evaluator:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"Check '{args.output_folder}' for detailed results.")
        return 0
    else:
        print(f"\n❌ Evaluation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 