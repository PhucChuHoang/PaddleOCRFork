#!/usr/bin/env python3
"""
OCR Accuracy Evaluation Script
Based on word-level anchor-based alignment from align.py

This script evaluates OCR model accuracy using standard metrics:
- Top-1 Word Accuracy (exact matches)
- Character Accuracy 
- CER (Character Error Rate)
- WER (Word Error Rate)
- Confidence-based analysis

Usage:
    python evaluate_ocr_accuracy.py --images_folder path/to/images --texts_folder path/to/ground_truth --output_folder path/to/results
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
import editdistance
import difflib

# Import functions from align.py
from align import (
    process_sequential_sentence_alignment,
    find_matching_files,
    load_ground_truth_file,
    load_nom_dictionary
)

class OCRAccuracyEvaluator:
    """OCR accuracy evaluation system with standard metrics."""
    
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
            'correct_words': 0,  # Exact matches only
        }
        
        self.character_metrics = {
            'total_characters': 0,
            'correct_characters': 0,
            'total_edit_distance': 0
        }
        
        self.edit_operations = {
            'substitutions': 0,
            'insertions': 0,
            'deletions': 0
        }
        
        self.confidence_metrics = {
            'high_confidence_correct': 0,
            'high_confidence_total': 0,
            'low_confidence_correct': 0,
            'low_confidence_total': 0,
        }
        
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
            'correct_words': 0,
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
            word_detail = result.get('word_detail')
            
            # Exact match only for word accuracy
            is_correct = (ocr_word == reference_word)
            if is_correct:
                file_word_metrics['correct_words'] += 1
            
            # Character-level metrics
            if ocr_word and reference_word:
                edit_distance = get_edit_distance(reference_word, ocr_word)
                max_len = max(len(ocr_word), len(reference_word))
                
                file_char_metrics['total_edit_distance'] += edit_distance
                file_char_metrics['total_characters'] += max_len
                file_char_metrics['correct_characters'] += max_len - edit_distance
            
            # Confidence analysis
            if word_detail:
                confidence = word_detail['confidence']
                confidence_data.append((confidence, is_correct))
        
        # Calculate file-level accuracy rates
        file_accuracy_metrics = {
            'top1_word_accuracy': file_word_metrics['correct_words'] / max(file_word_metrics['total_words'], 1) * 100,
            'character_accuracy': file_char_metrics['correct_characters'] / max(file_char_metrics['total_characters'], 1) * 100,
            'cer': file_char_metrics['total_edit_distance'] / max(file_char_metrics['total_characters'], 1),
            'wer': 1 - (file_word_metrics['correct_words'] / max(file_word_metrics['total_words'], 1))
        }
        
        if debug:
            print(f"\nFile Metrics for {file_name}:")
            print(f"  Top-1 Word Accuracy: {file_accuracy_metrics['top1_word_accuracy']:.2f}%")
            print(f"  Character Accuracy: {file_accuracy_metrics['character_accuracy']:.2f}%")
            print(f"  CER: {file_accuracy_metrics['cer']:.3f}")
            print(f"  WER: {file_accuracy_metrics['wer']:.3f}")
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
            word_detail = result.get('word_detail')
            
            # Exact match only for word accuracy
            is_correct = (ocr_word == reference_word)
            if is_correct:
                self.word_metrics['correct_words'] += 1
            
            # Character-level metrics and edit operations
            if ocr_word and reference_word:
                subs, ins, dels = get_edit_operations(reference_word, ocr_word)
                total_ops = subs + ins + dels
                max_len = max(len(ocr_word), len(reference_word))
                
                self.character_metrics['total_edit_distance'] += total_ops
                self.character_metrics['total_characters'] += max_len
                self.character_metrics['correct_characters'] += max_len - total_ops
                
                # Track edit operations
                self.edit_operations['substitutions'] += subs
                self.edit_operations['insertions'] += ins
                self.edit_operations['deletions'] += dels
            
            # Confidence analysis
            if word_detail:
                confidence = word_detail['confidence']
                
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
        """Calculate overall accuracy metrics using standard terminology."""
        
        top1_word_accuracy = self.word_metrics['correct_words'] / max(self.word_metrics['total_words'], 1) * 100
        character_accuracy = self.character_metrics['correct_characters'] / max(self.character_metrics['total_characters'], 1) * 100
        cer = self.character_metrics['total_edit_distance'] / max(self.character_metrics['total_characters'], 1)
        wer = 1 - (top1_word_accuracy / 100)
        
        overall_metrics = {
            'top1_word_accuracy': top1_word_accuracy,
            'character_accuracy': character_accuracy,
            'cer': cer,  # Character Error Rate
            'wer': wer,  # Word Error Rate
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
        
        # Edit operation rates
        total_words = max(self.word_metrics['total_words'], 1)
        overall_metrics['substitution_rate'] = self.edit_operations['substitutions'] / total_words
        overall_metrics['insertion_rate'] = self.edit_operations['insertions'] / total_words
        overall_metrics['deletion_rate'] = self.edit_operations['deletions'] / total_words
        
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
        print(f"Top-1 Word Accuracy: {overall_metrics['top1_word_accuracy']:.2f}%")
        print(f"Character Accuracy: {overall_metrics['character_accuracy']:.2f}%")
        print(f"CER (Character Error Rate): {overall_metrics['cer']:.3f} ({overall_metrics['cer']*100:.1f}%)")
        print(f"WER (Word Error Rate): {overall_metrics['wer']:.3f} ({overall_metrics['wer']*100:.1f}%)")
        print(f"Edit Rates - Sub: {overall_metrics['substitution_rate']:.3f}, Ins: {overall_metrics['insertion_rate']:.3f}, Del: {overall_metrics['deletion_rate']:.3f}")
        print(f"Total Files Processed: {len(self.file_results)}")
        print(f"Total Words Evaluated: {self.word_metrics['total_words']}")
        print(f"\nDetailed report saved in: {output_folder}")
    
    def generate_text_report(self, overall_metrics, output_folder):
        """Generate detailed text report."""
        
        report_path = os.path.join(output_folder, 'ocr_accuracy_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("OCR ACCURACY EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MAIN METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Top-1 Word Accuracy: {overall_metrics['top1_word_accuracy']:.2f}%\n")
            f.write(f"Character Accuracy: {overall_metrics['character_accuracy']:.2f}%\n")
            f.write(f"CER (Character Error Rate): {overall_metrics['cer']:.3f} ({overall_metrics['cer']*100:.1f}%)\n")
            f.write(f"WER (Word Error Rate): {overall_metrics['wer']:.3f} ({overall_metrics['wer']*100:.1f}%)\n\n")
            
            f.write("EDIT OPERATION RATES:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Substitution Rate: {overall_metrics['substitution_rate']:.3f} ({overall_metrics['substitution_rate']*100:.1f}%)\n")
            f.write(f"Insertion Rate: {overall_metrics['insertion_rate']:.3f} ({overall_metrics['insertion_rate']*100:.1f}%)\n")
            f.write(f"Deletion Rate: {overall_metrics['deletion_rate']:.3f} ({overall_metrics['deletion_rate']*100:.1f}%)\n\n")
            
            f.write("EDIT OPERATION COUNTS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Substitutions: {self.edit_operations['substitutions']}\n")
            f.write(f"Insertions: {self.edit_operations['insertions']}\n")
            f.write(f"Deletions: {self.edit_operations['deletions']}\n")
            f.write(f"Total Edit Operations: {sum(self.edit_operations.values())}\n\n")
            
            f.write("CONFIDENCE ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"High Confidence Accuracy: {overall_metrics['high_confidence_accuracy']:.2f}%\n")
            f.write(f"Low Confidence Accuracy: {overall_metrics['low_confidence_accuracy']:.2f}%\n")
            f.write(f"High Confidence Samples: {self.confidence_metrics['high_confidence_total']}\n")
            f.write(f"Low Confidence Samples: {self.confidence_metrics['low_confidence_total']}\n\n")
            
            f.write("WORD-LEVEL STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Words Processed: {self.word_metrics['total_words']}\n")
            f.write(f"Correct Words (Exact Matches): {self.word_metrics['correct_words']}\n\n")
            
            f.write("CHARACTER-LEVEL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Characters: {self.character_metrics['total_characters']}\n")
            f.write(f"Correct Characters: {self.character_metrics['correct_characters']}\n")
            f.write(f"Total Edit Distance: {self.character_metrics['total_edit_distance']}\n\n")
            
            f.write("PER-FILE RESULTS:\n")
            f.write("-" * 20 + "\n")
            for file_result in self.file_results:
                f.write(f"\nFile: {file_result['file_name']}\n")
                f.write(f"  Top-1 Word Accuracy: {file_result['accuracy_metrics']['top1_word_accuracy']:.2f}%\n")
                f.write(f"  Character Accuracy: {file_result['accuracy_metrics']['character_accuracy']:.2f}%\n")
                f.write(f"  CER: {file_result['accuracy_metrics']['cer']:.3f}\n")
                f.write(f"  WER: {file_result['accuracy_metrics']['wer']:.3f}\n")
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
            top1_acc = file_result['accuracy_metrics']['top1_word_accuracy']
            char_acc = file_result['accuracy_metrics']['character_accuracy']
            print(f"✅ {base_name}: Top1 {top1_acc:.1f}%, Char {char_acc:.1f}%")
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


def get_edit_distance(ref_word, ocr_word):
    """
    Get edit distance between two words.
    
    Parameters:
        ref_word: Reference (ground truth) word
        ocr_word: OCR predicted word
        
    Returns:
        int: Edit distance
    """
    return editdistance.eval(ref_word, ocr_word)


def get_edit_operations(ref_word, ocr_word):
    """
    Get the breakdown of edit operations (substitutions, insertions, deletions).
    
    Parameters:
        ref_word: Reference (ground truth) word
        ocr_word: OCR predicted word
        
    Returns:
        tuple: (substitutions, insertions, deletions)
    """
    # Use difflib.SequenceMatcher to get operation codes
    matcher = difflib.SequenceMatcher(None, ref_word, ocr_word)
    operations = matcher.get_opcodes()
    
    subs = sum(1 for op in operations if op[0] == 'replace')
    ins = sum(1 for op in operations if op[0] == 'insert') 
    dels = sum(1 for op in operations if op[0] == 'delete')
    return subs, ins, dels

if __name__ == "__main__":
    sys.exit(main()) 