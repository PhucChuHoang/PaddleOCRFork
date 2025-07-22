#!/usr/bin/env python3
"""
OCR Accuracy Evaluation Script (WORD-LEVEL VERSION)

Metrics:
- Top-1 Word Accuracy (AMR) = (N - (S + D)) / N
- WER = (S + I + D) / N   (word-level Levenshtein)
- Sub / Ins / Del rates = S/N, I/N, D/N (sum exactly to WER)
- Macro-F1 over word types (optional)
- Confidence vs. accuracy curve
- Confusion matrix for top-30 ground-truth word types

Character-level CER/accuracy removed per request (pure word-level evaluation).
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Project-specific imports
from align import (
    process_sequential_sentence_alignment,
    find_matching_files,
    load_ground_truth_file,
    load_nom_dictionary
)


# ---------------------------------------------------------------------------
# Word-level Levenshtein decomposition
# ---------------------------------------------------------------------------

def word_levenshtein_breakdown(ref_seq, hyp_seq):
    """
    Compute word-level Levenshtein counts (S, I, D) and distance.
    ref_seq: list of reference words
    hyp_seq: list of predicted words
    Returns: (S, I, D)
    """
    n, m = len(ref_seq), len(hyp_seq)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i  # deletions
    for j in range(1, m+1):
        dp[0][j] = j  # insertions

    for i in range(1, n+1):
        r_w = ref_seq[i-1]
        for j in range(1, m+1):
            h_w = hyp_seq[j-1]
            cost = 0 if r_w == h_w else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # match / substitution
            )

    # Backtrace to count ops
    i, j = n, m
    S = I = D = 0
    while i > 0 or j > 0:
        # Match or substitution
        if i > 0 and j > 0 and \
           dp[i][j] == dp[i-1][j-1] + (0 if ref_seq[i-1] == hyp_seq[j-1] else 1):
            if ref_seq[i-1] != hyp_seq[j-1]:
                S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, I, D


class OCRAccuracyEvaluator:
    """Word-level OCR accuracy evaluation."""

    def __init__(self, is_vietnamese=True, min_anchor_similarity=0.9, similarity_threshold=0.6):
        self.is_vietnamese = is_vietnamese
        self.min_anchor_similarity = min_anchor_similarity
        self.similarity_threshold = similarity_threshold
        self.nom_dict = load_nom_dictionary() if is_vietnamese else None
        self.reset_metrics()

    def reset_metrics(self):
        # Global word-level counts
        self.global_counts = {
            'N': 0,  # total reference words
            'S': 0,
            'I': 0,
            'D': 0,
            'Matches': 0,  # convenience
        }
        # For macro-F1/confusion
        self.gt_words = []
        self.pred_words = []
        # Confidence
        self.confidence_scores = []
        self.is_correct_flags = []
        # Per-file raw results
        self.file_results = []

    def evaluate_single_file(self, image_path, text_path, debug=False):
        reference_texts = load_ground_truth_file(text_path)
        if not reference_texts:
            print(f"Warning: No ground truth found for {text_path}")
            return None

        base_name = Path(image_path).stem
        if debug:
            print(f"\n=== EVALUATING: {base_name} ===")

        original_result, clustered_result, aligned_results = process_sequential_sentence_alignment(
            img_path=image_path,
            reference_texts=reference_texts,
            threshold=self.similarity_threshold,
            is_vertical=True,
            visualize=False,
            debug=debug,
        )

        file_metrics = self.calculate_file_metrics(aligned_results, base_name, debug)
        self.update_accumulated_metrics(aligned_results)
        self.file_results.append(file_metrics)
        return file_metrics

    def calculate_file_metrics(self, aligned_results, file_name, debug=False):
        """
        Compute per-file word-level S/I/D and derived metrics.
        aligned_results: list of dicts each containing 'ocr_word', 'reference_word', 'word_detail' (optional)
        """
        ref_seq = []
        hyp_seq = []
        confidences = []
        correct_flags = []

        for r in aligned_results:
            ref_w = r.get('reference_word', '')
            hyp_w = r.get('ocr_word', '')
            ref_seq.append(ref_w)
            hyp_seq.append(hyp_w)

            # Confidence
            wd = r.get('word_detail')
            if wd:
                confidences.append(wd['confidence'])
            # Correct flag for calibration plotting
            correct_flags.append(int(ref_w == hyp_w))

        S, I, D = word_levenshtein_breakdown(ref_seq, hyp_seq)
        N = len(ref_seq)
        wer = (S + I + D) / N if N else 0.0
        amr = (N - (S + D)) / N * 100 if N else 0.0  # Top-1 exact word accuracy

        file_metrics = {
            'file_name': file_name,
            'N': N,
            'S': S,
            'I': I,
            'D': D,
            'WER': wer,
            'AMR': amr,
            'sub_rate': S / N if N else 0.0,
            'ins_rate': I / N if N else 0.0,
            'del_rate': D / N if N else 0.0,
            'confidences': confidences,
            'correct_flags': correct_flags,
            'ref_seq': ref_seq,
            'hyp_seq': hyp_seq
        }

        if debug:
            print(f"File {file_name}: AMR={amr:.2f}% WER={wer:.3f} S={S} I={I} D={D} N={N}")

        return file_metrics

    def update_accumulated_metrics(self, aligned_results):
        # Accumulate global sequences for macro-F1/confusion and confidence
        ref_seq = []
        hyp_seq = []
        for r in aligned_results:
            ref_w = r.get('reference_word', '')
            hyp_w = r.get('ocr_word', '')
            ref_seq.append(ref_w)
            hyp_seq.append(hyp_w)
            self.gt_words.append(ref_w)
            self.pred_words.append(hyp_w)

            wd = r.get('word_detail')
            if wd:
                conf = wd['confidence']
                self.confidence_scores.append(conf)
                self.is_correct_flags.append(int(ref_w == hyp_w))

        # After we append for this file, accumulate S/I/D once
        S, I, D = word_levenshtein_breakdown(ref_seq, hyp_seq)
        N = len(ref_seq)
        self.global_counts['S'] += S
        self.global_counts['I'] += I
        self.global_counts['D'] += D
        self.global_counts['N'] += N
        self.global_counts['Matches'] += (N - (S + D))

    def calculate_overall_accuracy(self):
        S = self.global_counts['S']
        I = self.global_counts['I']
        D = self.global_counts['D']
        N = self.global_counts['N']
        matches = self.global_counts['Matches']

        if N == 0:
            return {}

        wer = (S + I + D) / N
        amr = matches / N * 100.0  # Word accuracy %
        sub_rate = S / N
        ins_rate = I / N
        del_rate = D / N  # sub + ins + del == wer

        # Macro-F1 over word labels (optional)
        macro_f1 = 0.0
        if self.gt_words:
            try:
                macro_f1 = f1_score(self.gt_words, self.pred_words, average='macro')
            except Exception as e:
                print(f"Warning (macro-F1): {e}")

        metrics = {
            'top1_word_accuracy': amr,
            'wer': wer,
            'sub_rate': sub_rate,
            'ins_rate': ins_rate,
            'del_rate': del_rate,
            'S': S, 'I': I, 'D': D, 'N': N,
            'macro_f1': macro_f1,
            'average_confidence': (sum(self.confidence_scores)/len(self.confidence_scores)
                                   if self.confidence_scores else 0.0)
        }
        return metrics

    # ---------------- Confidence Calibration ---------------- #

    def binned_accuracy(self, confidence_scores, correct_flags, bins):
        if not confidence_scores:
            return [], [], []
        cs = np.array(confidence_scores)
        cf = np.array(correct_flags)
        idx = np.digitize(cs, bins) - 1
        centers, accs, counts = [], [], []
        for i in range(len(bins)-1):
            mask = idx == i
            if mask.any():
                centers.append((bins[i]+bins[i+1])/2)
                accs.append(cf[mask].mean())
                counts.append(mask.sum())
        return centers, accs, counts

    def plot_confidence_accuracy_curve(self, output_folder):
        if not self.confidence_scores:
            print("No confidence data to plot.")
            return
        bins = np.arange(0, 1.05, 0.05)
        centers, accs, counts = self.binned_accuracy(self.confidence_scores, self.is_correct_flags, bins)
        if not centers:
            print("No valid bins for calibration curve.")
            return

        plt.figure(figsize=(8,6))
        plt.plot(centers, accs, 'bo-', label='Empirical Accuracy', linewidth=2)
        plt.plot([0,1],[0,1],'r--',label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Confidence vs Accuracy')
        plt.xlim(0,1); plt.ylim(0,1)
        plt.grid(alpha=0.3)
        for x,y,c in zip(centers, accs, counts):
            plt.annotate(f"n={c}", (x,y), textcoords='offset points', xytext=(4,4), fontsize=8)
        plt.legend()
        path = os.path.join(output_folder, 'confidence_accuracy_curve.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        # Save data
        with open(os.path.join(output_folder,'confidence_accuracy_data.txt'),'w',encoding='utf-8') as f:
            f.write("center\taccuracy\tcount\n")
            for x,y,c in zip(centers, accs, counts):
                f.write(f"{x:.3f}\t{y:.3f}\t{c}\n")
        print(f"Confidence-accuracy curve saved to {path}")

    # ---------------- Confusion Matrix ---------------- #

    def generate_confusion_matrix(self, output_folder, plot_confusion=False):
        if not self.gt_words:
            print("No words for confusion matrix.")
            return
        top30 = [w for w,_ in Counter(self.gt_words).most_common(30)]
        cm = confusion_matrix(self.gt_words, self.pred_words, labels=top30)
        # Save text
        cm_path = os.path.join(output_folder, 'confusion_matrix.txt')
        with open(cm_path,'w',encoding='utf-8') as f:
            f.write("CONFUSION MATRIX (Top-30 GT words)\n\n")
            for i,w in enumerate(top30):
                f.write(f"{i:2d}: {w}\n")
            f.write("\n")
            np.savetxt(f, cm, fmt='%d')
        print(f"Confusion matrix saved to {cm_path}")

        if plot_confusion:
            try:
                plt.figure(figsize=(12,10))
                sns.heatmap(cm, xticklabels=top30, yticklabels=top30,
                            annot=True, fmt='d', cmap='Blues', cbar_kws={'label':'Count'})
                plt.title('Confusion Matrix (Top-30)')
                plt.xlabel('Predicted')
                plt.ylabel('Ground Truth')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(output_folder,'confusion.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"Confusion matrix plot saved to {plot_path}")
            except Exception as e:
                print(f"Plot confusion error: {e}")

    # ---------------- Reporting ---------------- #

    def generate_text_report(self, metrics, output_folder):
        out = os.path.join(output_folder,'ocr_accuracy_report.txt')
        with open(out,'w',encoding='utf-8') as f:
            f.write("WORD-LEVEL OCR EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write("GLOBAL METRICS:\n")
            f.write(f"Reference Words (N): {metrics['N']}\n")
            f.write(f"S / I / D: {metrics['S']} / {metrics['I']} / {metrics['D']}\n")
            f.write(f"WER: {metrics['wer']:.3f} ({metrics['wer']*100:.1f}%)\n")
            f.write(f"Top-1 Word Accuracy (AMR): {metrics['top1_word_accuracy']:.2f}%\n")
            f.write(f"Sub Rate: {metrics['sub_rate']:.3f}  Ins Rate: {metrics['ins_rate']:.3f}  Del Rate: {metrics['del_rate']:.3f}\n")
            f.write(f"Check sum (should equal WER): {(metrics['sub_rate']+metrics['ins_rate']+metrics['del_rate']):.3f}\n")
            f.write(f"Macro-F1 (word types): {metrics['macro_f1']:.3f}\n")
            f.write(f"Average Confidence: {metrics['average_confidence']:.3f}\n\n")

            f.write("PER-FILE SUMMARY:\n")
            for fr in self.file_results:
                f.write(f"- {fr['file_name']}: AMR={fr['AMR']:.2f}% WER={fr['WER']:.3f} "
                        f"S={fr['S']} I={fr['I']} D={fr['D']} N={fr['N']}\n")
        print(f"Text report saved to {out}")

    def generate_evaluation_report(self, output_folder, plot_confusion=False):
        os.makedirs(output_folder, exist_ok=True)
        metrics = self.calculate_overall_accuracy()
        if not metrics:
            print("No data to report.")
            return

        # Plots
        self.plot_confidence_accuracy_curve(output_folder)
        self.generate_confusion_matrix(output_folder, plot_confusion)
        self.generate_text_report(metrics, output_folder)

        print("\n=== EVALUATION REPORT GENERATED ===")
        print(f"AMR: {metrics['top1_word_accuracy']:.2f}%  WER: {metrics['wer']:.3f}")
        print(f"S/I/D: {metrics['S']}/{metrics['I']}/{metrics['D']}  N={metrics['N']}")
        print(f"Macro-F1: {metrics['macro_f1']:.3f}  AvgConf: {metrics['average_confidence']:.3f}")
        print("Report artifacts written to:", output_folder)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def evaluate_batch_accuracy(images_folder, texts_folder, output_folder,
                            min_anchor_similarity=0.9, similarity_threshold=0.6,
                            is_vietnamese=True, plot_confusion=False, debug=False):

    evaluator = OCRAccuracyEvaluator(
        is_vietnamese=is_vietnamese,
        min_anchor_similarity=min_anchor_similarity,
        similarity_threshold=similarity_threshold
    )

    pairs = find_matching_files(images_folder, texts_folder)
    if not pairs:
        print("No matching image-text pairs.")
        return None

    print(f"\n=== OCR ACCURACY EVALUATION: {len(pairs)} FILES ===")
    print(f"Images: {images_folder}")
    print(f"Ground truth: {texts_folder}")
    print(f"Output: {output_folder}")

    ok = 0
    for idx, (img_path, txt_path, base_name) in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] {base_name}")
        res = evaluator.evaluate_single_file(
            img_path, txt_path, debug=(debug and idx <= 3)
        )
        if res:
            ok += 1
            print(f"  ✔ AMR {res['AMR']:.1f}%  WER {res['WER']:.3f}")
        else:
            print("  ✖ (skipped)")

    print(f"\nProcessed: {ok}/{len(pairs)} successful.")
    if ok:
        evaluator.generate_evaluation_report(output_folder, plot_confusion)
    return evaluator


def main():
    parser = argparse.ArgumentParser(description="Word-level OCR accuracy evaluation")
    parser.add_argument('--images_folder', required=True)
    parser.add_argument('--texts_folder', required=True)
    parser.add_argument('--output_folder', default='evaluation_results')
    parser.add_argument('--anchor_threshold', type=float, default=0.9)
    parser.add_argument('--similarity_threshold', type=float, default=0.6)
    parser.add_argument('--is_vietnamese', action='store_true', default=True)
    parser.add_argument('--plot_confusion', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.images_folder):
        print(f"Images folder not found: {args.images_folder}")
        return 1
    if not os.path.isdir(args.texts_folder):
        print(f"Texts folder not found: {args.texts_folder}")
        return 1

    evaluate_batch_accuracy(
        images_folder=args.images_folder,
        texts_folder=args.texts_folder,
        output_folder=args.output_folder,
        min_anchor_similarity=args.anchor_threshold,
        similarity_threshold=args.similarity_threshold,
        is_vietnamese=args.is_vietnamese,
        plot_confusion=args.plot_confusion,
        debug=args.debug
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
