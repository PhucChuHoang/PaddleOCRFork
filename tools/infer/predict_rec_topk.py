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
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import math
import time
import traceback
import paddle

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


class TextRecognizerTopK(object):
    def __init__(self, args, logger=None):
        if os.path.exists(f"{args.rec_model_dir}/inference.yml"):
            model_config = utility.load_config(f"{args.rec_model_dir}/inference.yml")
            print("model_config", model_config)
            model_name = model_config.get("Global", {}).get("model_name", "")
            if model_name:
                raise ValueError(
                    f"{model_name} is not supported. Please check if the model is supported by the PaddleOCR wheel."
                )

            if args.rec_char_dict_path == "./ppocr/utils/ppocr_keys_v1.txt":
                rec_char_list = model_config.get("PostProcess", {}).get(
                    "character_dict", []
                )
                if rec_char_list:
                    new_rec_char_dict_path = f"{args.rec_model_dir}/ppocr_keys.txt"
                    with open(new_rec_char_dict_path, "w", encoding="utf-8") as f:
                        f.writelines([char + "\n" for char in rec_char_list])
                    args.rec_char_dict_path = new_rec_char_dict_path

        if logger is None:
            logger = get_logger()
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.k = args.k  # Number of top-k results to return
        
        # Use CTCLabelDecodeTopK for SVTR algorithm
        if self.rec_algorithm == "SVTR":
            postprocess_params = {
                "name": "CTCLabelDecodeTopK",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "k": self.k,
            }
        elif self.rec_algorithm == "SRN":
            postprocess_params = {
                "name": "SRNLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "RARE":
            postprocess_params = {
                "name": "AttnLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "NRTR":
            postprocess_params = {
                "name": "NRTRLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "SAR":
            postprocess_params = {
                "name": "SARLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "VisionLAN":
            postprocess_params = {
                "name": "VLLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "max_text_length": args.max_text_length,
            }
        elif self.rec_algorithm == "ViTSTR":
            postprocess_params = {
                "name": "ViTSTRLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "ABINet":
            postprocess_params = {
                "name": "ABINetLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "SPIN":
            postprocess_params = {
                "name": "SPINLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "RobustScanner":
            postprocess_params = {
                "name": "SARLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True,
            }
        elif self.rec_algorithm == "RFL":
            postprocess_params = {
                "name": "RFLLabelDecode",
                "character_dict_path": None,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "SATRN":
            postprocess_params = {
                "name": "SATRNLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True,
            }
        elif self.rec_algorithm in ["CPPD", "CPPDPadding"]:
            postprocess_params = {
                "name": "CPPDLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True,
            }
        elif self.rec_algorithm == "PREN":
            postprocess_params = {"name": "PRENLabelDecode"}
        elif self.rec_algorithm == "CAN":
            self.inverse = args.rec_image_inverse
            postprocess_params = {
                "name": "CANLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "LaTeXOCR":
            postprocess_params = {
                "name": "LaTeXOCRDecode",
                "rec_char_dict_path": args.rec_char_dict_path,
            }
        elif self.rec_algorithm == "ParseQ":
            postprocess_params = {
                "name": "ParseQLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        else:
            # Default to CTCLabelDecodeTopK for other algorithms
            postprocess_params = {
                "name": "CTCLabelDecodeTopK",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "k": self.k,
            }
            
        self.postprocess_op = build_post_process(postprocess_params)
        self.postprocess_params = postprocess_params
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = utility.create_predictor(args, "rec", logger)
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx
        if args.benchmark:
            import auto_log

            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="rec",
                model_precision=args.precision,
                batch_size=args.rec_batch_num,
                data_shape="dynamic",
                save_path=None,  # not used if logger is not None
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=["preprocess_time", "inference_time", "postprocess_time"],
                warmup=0,
                logger=logger,
            )
        self.return_word_box = args.return_word_box

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.rec_algorithm == "NRTR" or self.rec_algorithm == "ViTSTR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # return padding_im
            image_pil = Image.fromarray(np.uint8(img))
            if self.rec_algorithm == "ViTSTR":
                img = image_pil.resize([imgW, imgH], Image.BICUBIC)
            else:
                img = image_pil.resize([imgW, imgH], Image.Resampling.LANCZOS)
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            if self.rec_algorithm == "ViTSTR":
                norm_img = norm_img.astype(np.float32) / 255.0
            else:
                norm_img = norm_img.astype(np.float32) / 128.0 - 1.0
            return norm_img
        elif self.rec_algorithm == "RFL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype("float32")
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if isinstance(w, str):
                pass
            elif w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        if self.rec_algorithm == "RARE":
            if resized_w > self.rec_image_shape[2]:
                resized_w = self.rec_image_shape[2]
            imgW = self.rec_image_shape[2]
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_vl(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        img = cv2.resize(img, (imgW, imgH))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        return img

    def resize_norm_img_svtr(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [[] for _ in range(img_num)]  # Initialize with empty lists for top-k results
        batch_num = self.rec_batch_num
        st = time.time()
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == "SVTR":
                    norm_img = self.resize_norm_img_svtr(
                        img_list[indices[ino]], self.rec_image_shape
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                else:
                    norm_img = self.resize_norm_img(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()

            if self.use_onnx:
                input_dict = {}
                input_dict[self.input_tensor.name] = norm_img_batch
                outputs = self.predictor.run(self.output_tensors, input_dict)
                preds = outputs[0]
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()
                outputs = []
                for output_tensor in self.output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                if self.benchmark:
                    self.autolog.times.stamp()
                if len(outputs) != 1:
                    preds = outputs
                else:
                    preds = outputs[0]
                    
            if self.postprocess_params["name"] == "CTCLabelDecodeTopK":
                rec_result = self.postprocess_op(
                    preds,
                    return_word_box=self.return_word_box,
                    wh_ratio_list=wh_ratio_list,
                    max_wh_ratio=max_wh_ratio,
                )
            else:
                rec_result = self.postprocess_op(preds)
                
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if self.benchmark:
                self.autolog.times.end(stamp=True)
        return rec_res, time.time() - st


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    valid_image_file_list = []
    img_list = []

    # logger
    log_file = args.save_log_path
    if os.path.isdir(args.save_log_path) or (
        not os.path.exists(args.save_log_path) and args.save_log_path.endswith("/")
    ):
        log_file = os.path.join(log_file, "benchmark_recognition_topk.log")
    logger = get_logger(log_file=log_file)

    # create text recognizer with top-k
    text_recognizer = TextRecognizerTopK(args)

    logger.info(
        "Using top-k text recognition with k=%d. Make sure you're using a model with SVTR algorithm.", args.k
    )

    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    for ino in range(len(img_list)):
        logger.info(
            "Predicts of {}:".format(valid_image_file_list[ino])
        )
        # Print all top-k results for each image
        for idx, res in enumerate(rec_res[ino]):
            logger.info("  [{}] {}, confidence: {:.4f}".format(idx+1, res[0], res[1]))
    if args.benchmark:
        text_recognizer.autolog.report()


if __name__ == "__main__":
    # Add k parameter to the command line arguments
    args = utility.parse_args()
    args.k = 5  # Default value if not specified
    
    # Check if k is already in args
    if not hasattr(args, 'k'):
        import argparse
        parser = argparse.ArgumentParser(parents=[utility.get_arg_parser()])
        parser.add_argument('--k', type=int, default=5, help='Number of top-k results to return')
        args = parser.parse_args()
    
    main(args) 