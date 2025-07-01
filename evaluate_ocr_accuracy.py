#!/usr/bin/env python3
"""
OCR Accuracy Evaluation Script
Based on word-level anchor-based alignment from align.py

This script evaluates OCR model accuracy using standard metrics:
- Top-1 Word Accuracy (exact matches)
- Character Accuracy 
- CER (Character Error Rate)
- WER (Word Error Rate)
- Macro-F1 (treating each word as a class)
- Confusion matrix for top-30 most frequent confusions
- Confidence-based analysis

Usage:
    python evaluate_ocr_accuracy.py --images_folder path/to/images --texts_folder path/to/ground_truth --output_folder path/to/results
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import editdistance
import difflib

# Sklearn imports for macro-F1 and confusion matrix
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Simplified confidence analysis with binning
        self.confidence_scores = []  # All confidence scores
        self.is_correct_flags = []   # Corresponding correctness flags
        
        # Track words for macro-F1 and confusion matrix
        self.gt_words = []  # Ground truth words
        self.pred_words = []  # Predicted words
        
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
            
            # Confidence data collection
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
            
            # Track words for macro-F1 and confusion matrix
            self.gt_words.append(reference_word)
            self.pred_words.append(ocr_word)
            
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
                
                # Track confidence scores and correctness flags
                self.confidence_scores.append(confidence)
                self.is_correct_flags.append(is_correct)
    
    def calculate_overall_accuracy(self):
        """Calculate overall accuracy metrics using standard terminology."""
        
        top1_word_accuracy = self.word_metrics['correct_words'] / max(self.word_metrics['total_words'], 1) * 100
        character_accuracy = self.character_metrics['correct_characters'] / max(self.character_metrics['total_characters'], 1) * 100
        cer = self.character_metrics['total_edit_distance'] / max(self.character_metrics['total_characters'], 1)
        wer = 1 - (top1_word_accuracy / 100)
        
        # Calculate macro-F1 (treating each word as a class)
        macro_f1 = 0.0
        if len(self.gt_words) > 0 and len(self.pred_words) > 0:
            try:
                macro_f1 = f1_score(self.gt_words, self.pred_words, average='macro')
            except Exception as e:
                print(f"Warning: Could not calculate macro-F1: {e}")
                macro_f1 = 0.0
        
        overall_metrics = {
            'top1_word_accuracy': top1_word_accuracy,
            'character_accuracy': character_accuracy,
            'cer': cer,  # Character Error Rate
            'wer': wer,  # Word Error Rate
            'macro_f1': macro_f1,  # Macro-F1 score
        }
        
        # Confidence-based metrics
        if len(self.confidence_scores) > 0:
            overall_metrics['average_confidence'] = sum(self.confidence_scores) / len(self.confidence_scores)
        else:
            overall_metrics['average_confidence'] = 0
            
        # Edit operation rates
        total_words = max(self.word_metrics['total_words'], 1)
        overall_metrics['substitution_rate'] = self.edit_operations['substitutions'] / total_words
        overall_metrics['insertion_rate'] = self.edit_operations['insertions'] / total_words
        overall_metrics['deletion_rate'] = self.edit_operations['deletions'] / total_words
        
        return overall_metrics
    
    def binned_accuracy(self, confidence_scores, is_correct_flags, bins):
        """
        Calculate accuracy for each confidence bin.
        
        Parameters:
            confidence_scores: List of confidence scores
            is_correct_flags: List of correctness flags
            bins: Bin edges for confidence scores
            
        Returns:
            tuple: (bin_centers, accuracy_per_bin, counts_per_bin)
        """
        if len(confidence_scores) == 0:
            return [], [], []
            
        confidence_scores = np.array(confidence_scores)
        is_correct_flags = np.array(is_correct_flags)
        
        # Digitize confidence scores into bins
        bin_indices = np.digitize(confidence_scores, bins) - 1
        
        bin_centers = []
        accuracy_per_bin = []
        counts_per_bin = []
        
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_center = (bins[i] + bins[i + 1]) / 2
                accuracy = is_correct_flags[mask].mean()
                count = mask.sum()
                
                bin_centers.append(bin_center)
                accuracy_per_bin.append(accuracy)
                counts_per_bin.append(count)
        
        return bin_centers, accuracy_per_bin, counts_per_bin
    
    def plot_confidence_accuracy_curve(self, output_folder):
        """
        Plot accuracy-vs-confidence curve and save as PNG.
        
        Parameters:
            output_folder: Output directory to save the plot
        """
        if len(self.confidence_scores) == 0:
            print("Warning: No confidence data available for plotting")
            return
            
        try:
            # Define bins from 0 to 1.05 with step 0.05
            bins = np.arange(0, 1.05, 0.05)
            
            # Calculate binned accuracy
            bin_centers, accuracy_per_bin, counts_per_bin = self.binned_accuracy(
                self.confidence_scores, self.is_correct_flags, bins
            )
            
            if len(bin_centers) == 0:
                print("Warning: No valid bins found for confidence curve")
                return
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot accuracy vs confidence
            plt.plot(bin_centers, accuracy_per_bin, 'bo-', linewidth=2, markersize=8, label='Accuracy')
            
            # Add perfect calibration line (diagonal)
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
            
            # Customize the plot
            plt.xlabel('Confidence Score', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Accuracy vs Confidence Score', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            # Add text annotations for sample counts
            for i, (x, y, count) in enumerate(zip(bin_centers, accuracy_per_bin, counts_per_bin)):
                if i % 2 == 0:  # Annotate every other point to avoid clutter
                    plt.annotate(f'n={count}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_folder, 'confidence_accuracy_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confidence-accuracy curve saved to {plot_path}")
            
            # Also save binned data as text
            data_path = os.path.join(output_folder, 'confidence_accuracy_data.txt')
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write("CONFIDENCE vs ACCURACY ANALYSIS\n")
                f.write("=" * 40 + "\n\n")
                f.write("Bin Center\tAccuracy\tCount\n")
                f.write("-" * 30 + "\n")
                for center, acc, count in zip(bin_centers, accuracy_per_bin, counts_per_bin):
                    f.write(f"{center:.3f}\t\t{acc:.3f}\t\t{count}\n")
                    
            print(f"Confidence-accuracy data saved to {data_path}")
            
        except Exception as e:
            print(f"Error plotting confidence-accuracy curve: {e}")
    
    def generate_confusion_matrix(self, output_folder, plot_confusion=False):
        """
        Generate confusion matrix for the top-30 most frequent confusions.
        
        Parameters:
            output_folder: Output directory to save results
            plot_confusion: Whether to plot and save confusion matrix as PNG
            
        Returns:
            tuple: (confusion_matrix, top30_vocab) or None if insufficient data
        """
        if len(self.gt_words) == 0 or len(self.pred_words) == 0:
            print("Warning: No word data available for confusion matrix")
            return None
            
        # Get top-30 most frequent vocabulary from ground truth
        word_counts = Counter(self.gt_words)
        top30_words = [word for word, count in word_counts.most_common(30)]
        
        if len(top30_words) == 0:
            print("Warning: No frequent words found for confusion matrix")
            return None
        
        try:
            # Generate confusion matrix for top-30 vocabulary
            cm = confusion_matrix(self.gt_words, self.pred_words, labels=top30_words)
            
            # Save confusion matrix as text
            cm_path = os.path.join(output_folder, 'confusion_matrix.txt')
            with open(cm_path, 'w', encoding='utf-8') as f:
                f.write("CONFUSION MATRIX - TOP 30 MOST FREQUENT WORDS\n")
                f.write("=" * 60 + "\n\n")
                f.write("Labels (Ground Truth -> Predicted):\n")
                for i, word in enumerate(top30_words):
                    f.write(f"{i:2d}: {word}\n")
                f.write("\n")
                
                # Write matrix
                np.savetxt(f, cm, fmt='%d')
            
            print(f"Confusion matrix saved to {cm_path}")
            
            # Plot confusion matrix if requested
            if plot_confusion:
                self.plot_confusion_matrix(cm, top30_words, output_folder)
                
            return cm, top30_words
            
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return None
    
    def plot_confusion_matrix(self, cm, labels, output_folder):
        """
        Plot and save confusion matrix as PNG.
        
        Parameters:
            cm: Confusion matrix array
            labels: Label names for the matrix
            output_folder: Output directory to save the plot
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(cm, 
                       xticklabels=labels, 
                       yticklabels=labels,
                       annot=True, 
                       fmt='d', 
                       cmap='Blues',
                       cbar_kws={'label': 'Count'})
            
            plt.title('Confusion Matrix - Top 30 Most Frequent Words', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Words', fontsize=12)
            plt.ylabel('Ground Truth Words', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_folder, 'confusion.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix plot saved to {plot_path}")
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
    
    def generate_evaluation_report(self, output_folder, plot_confusion=False):
        """Generate comprehensive evaluation report."""
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_accuracy()
        
        # Generate confidence-accuracy curve (always generated)
        self.plot_confidence_accuracy_curve(output_folder)
        
        # Generate confusion matrix
        self.generate_confusion_matrix(output_folder, plot_confusion)
        
        # Generate text report
        self.generate_text_report(overall_metrics, output_folder)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT GENERATED")
        print(f"{'='*60}")
        print(f"Top-1 Word Accuracy: {overall_metrics['top1_word_accuracy']:.2f}%")
        print(f"Character Accuracy: {overall_metrics['character_accuracy']:.2f}%")
        print(f"CER (Character Error Rate): {overall_metrics['cer']:.3f} ({overall_metrics['cer']*100:.1f}%)")
        print(f"WER (Word Error Rate): {overall_metrics['wer']:.3f} ({overall_metrics['wer']*100:.1f}%)")
        print(f"Macro-F1: {overall_metrics['macro_f1']:.3f}")
        print(f"Average Confidence: {overall_metrics['average_confidence']:.3f}")
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
            f.write(f"WER (Word Error Rate): {overall_metrics['wer']:.3f} ({overall_metrics['wer']*100:.1f}%)\n")
            f.write(f"Macro-F1: {overall_metrics['macro_f1']:.3f}\n\n")
            
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
            f.write(f"Average Confidence: {overall_metrics['average_confidence']:.3f}\n")
            f.write(f"Total Words with Confidence: {len(self.confidence_scores)}\n")
            if len(self.confidence_scores) > 0:
                f.write(f"Min Confidence: {min(self.confidence_scores):.3f}\n")
                f.write(f"Max Confidence: {max(self.confidence_scores):.3f}\n")
                f.write("See confidence_accuracy_curve.png and confidence_accuracy_data.txt for detailed analysis\n\n")
            else:
                f.write("No confidence data available\n\n")
            
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
                          is_vietnamese=True, plot_confusion=False, debug=False):
    """
    Evaluate OCR accuracy for a batch of image-text pairs.
    
    Parameters:
        images_folder: Path to folder containing images
        texts_folder: Path to folder containing ground truth text files
        output_folder: Path to output folder for evaluation results
        min_anchor_similarity: Minimum similarity for anchor points
        similarity_threshold: Threshold for considering matches as accurate
        is_vietnamese: Whether to use Vietnamese-specific processing
        plot_confusion: Whether to plot and save confusion matrix
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
        evaluator.generate_evaluation_report(output_folder, plot_confusion)
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
    parser.add_argument('--plot_confusion', action='store_true',
                       help='Generate and save confusion matrix plot as PNG')
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
        plot_confusion=args.plot_confusion,
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