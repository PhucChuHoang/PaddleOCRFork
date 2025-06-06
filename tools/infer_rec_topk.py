# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import cv2
import numpy as np
import argparse
import time

from tools.infer.predict_rec_topk import TextRecognizerTopK
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger

logger = get_logger()

def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")

def get_sort_key(filename):
    # Extract column and object numbers from filename
    # Example: "column1_obj1.jpg" -> (1, 1)
    try:
        base = os.path.basename(filename)
        # Extract column number - handle both formats like "column1" and "column10"
        if 'column' in base:
            col_part = base.split('column')[1].split('_')[0]
            col_num = int(col_part)
        else:
            col_num = 999999  # High number for files without column designation
            
        # Extract object number - handle both formats like "obj1" and "obj10"
        if 'obj' in base:
            obj_part = base.split('obj')[1].split('.')[0]
            obj_num = int(obj_part)
        else:
            obj_num = 999999  # High number for files without object designation
            
        # Return tuple for sorting - first by column number, then by object number
        return (col_num, obj_num)
    except Exception as e:
        # Print debugging info but don't fail
        print(f"Sorting error for {filename}: {e}")
        return (999999, 999999)  # Default sort for files that don't match pattern

def char2code(ch):
    pos = ord(ch) - 0xF0000
    return pos

def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory of images to be recognized."
    )
    parser.add_argument(
        "--rec_model_dir", type=str, required=True, help="Directory of recognition inference model."
    )
    parser.add_argument(
        "--use_gcu",
        type=str2bool,
        default=False,
        help="Use Enflame GCU(General Compute Unit)",
    )
    
    # Optional parameters
    parser.add_argument(
        "--k", type=int, default=5, help="Number of top-k results to return"
    )
    parser.add_argument(
        "--rec_image_shape", type=str, default="3, 48, 48", help="Image shape for recognition model: 'C,H,W'"
    )
    parser.add_argument(
        "--rec_batch_num", type=int, default=6, help="Batch size for recognition"
    )
    parser.add_argument(
        "--rec_algorithm", type=str, default="SVTR", help="Recognition algorithm. SVTR works with top-k."
    )
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="ppocr/utils/dict/casia_hwdb_dict.txt",
        help="Character dictionary path"
    )
    parser.add_argument(
        "--use_space_char", type=bool, default=False, help="Whether to recognize space"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="Whether to use GPU"
    )
    parser.add_argument(
        "--gpu_mem", type=int, default=8000, help="GPU memory allocation (MB)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID for inference"
    )
    parser.add_argument(
        "--enable_mkldnn", type=bool, default=False, help="Whether to enable MKLDNN"
    )
    parser.add_argument(
        "--use_tensorrt", type=bool, default=False, help="Whether to use tensorrt"
    )
    parser.add_argument(
        "--precision", type=str, default="fp32", help="Precision type"
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./log_output/",
        help="Folder for saving logs"
    )
    parser.add_argument(
        "--max_text_length", type=int, default=25, help="Maximum text length"
    )
    parser.add_argument(
        "--use_onnx", action='store_true', help="Whether to use ONNX model"
    )
    parser.add_argument(
        "--benchmark", action='store_true', help="Whether to enable benchmark"
    )
    parser.add_argument(
        "--warmup", action='store_true', help="Whether to warmup before inference"
    )
    parser.add_argument(
        "--return_word_box", action='store_true', help="Whether to return word box"
    )
    parser.add_argument(
        "--rec_image_inverse", action='store_true', help="Whether to do image inverse in recognition"
    )
    parser.add_argument(
        "--output_file", type=str, default="recognition_results.txt", 
        help="File path to save the recognition results"
    )
    
    return parser.parse_args()

def write_results_to_file(output_file, image_file_list, rec_results, nom_dict):
    """Write recognition results to a text file in an LLM-friendly format"""
    # Group files by column for better organization
    columns = {}
    for idx, image_file in enumerate(image_file_list):
        # Extract column number from filename
        base = os.path.basename(image_file)
        try:
            if 'column' in base:
                col_part = base.split('column')[1].split('_')[0]
                col_num = int(col_part)
            else:
                col_num = 999999
        except:
            col_num = 999999
            
        if col_num not in columns:
            columns[col_num] = []
            
        columns[col_num].append((image_file, rec_results[idx]))
    
    # Write results in structured format
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# OCR Recognition Results\n\n")
        f.write("This document contains character recognition results organized by column.\n")
        f.write("For each character, multiple interpretations are provided with confidence scores.\n\n")
        
        # Sort columns numerically
        for col_num in sorted(columns.keys()):
            if col_num == 999999:
                f.write("## Uncategorized Images\n\n")
            else:
                f.write(f"## Column {col_num}\n\n")
            
            f.write("Characters in this column (from top to bottom):\n\n")
            
            # Sort by object number within column
            col_images = sorted(columns[col_num], key=lambda x: get_sort_key(x[0])[1])
            
            for img_file, results in col_images:
                base = os.path.basename(img_file)
                
                # Extract object number for reference
                obj_num = "unknown"
                if 'obj' in base:
                    try:
                        obj_part = base.split('obj')[1].split('.')[0]
                        obj_num = obj_part
                    except:
                        pass
                
                f.write(f"Character {obj_num}:\n")
                
                # List all interpretations
                interpretations = []
                for rank, result in enumerate(results):
                    text, confidence = result
                    try:
                        index = char2code(text) - 1
                        display_text = nom_dict[index] if index >= 0 and index < len(nom_dict) else text
                    except:
                        display_text = text
                    
                    interpretations.append(f"{display_text} ({confidence:.2f})")
                
                # Join interpretations into a sentence
                if interpretations:
                    f.write(f"This character could be: {', '.join(interpretations)}\n\n")
                else:
                    f.write("No interpretations available for this character.\n\n")
        
        f.write("End of recognition results.\n")
    
    logger.info(f"Results saved to {output_file} in LLM-friendly format")

def main():
    args = parse_args()
    
    with open('C:/Users/ADMIN/Downloads/new_dict.txt', 'r', encoding='utf-8') as f:
        nom_dict = f.read().splitlines()
    
    # Create text recognizer with top-k
    text_recognizer = TextRecognizerTopK(args)
    
    # Process images
    image_file_list = get_image_file_list(args.image_dir)

    # Sort the image files based on column and object numbers
    image_file_list.sort(key=get_sort_key)

    valid_image_file_list = []
    img_list = []
    
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
        
    try:
        start_time = time.time()
        rec_results, _ = text_recognizer(img_list)
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Processed {len(img_list)} images in {total_time:.3f} seconds ({len(img_list)/total_time:.2f} images/sec)")
        
        # Print results
        for idx, image_file in enumerate(valid_image_file_list):
            logger.info(f"\nImage: {image_file}")
            logger.info(f"Top-{len(rec_results[idx])} results:")
            for rank, result in enumerate(rec_results[idx]):
                text, confidence = result
                index = char2code(text) - 1
                logger.info(f"  [{rank+1}] Text: {nom_dict[index]}, Confidence: {confidence:.4f}")
        
        # Write results to file
        write_results_to_file(args.output_file, valid_image_file_list, rec_results, nom_dict)
                
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        logger.error(f"Traceback: {sys.exc_info()[0]}")
        sys.exit(1)

if __name__ == "__main__":
    main() 