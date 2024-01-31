#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
#

import os

import numpy as np

try:
    import tensorflow.lite as tflite
except:
    import tflite_runtime.interpreter as tflite


class NMSModel:
    def __init__(self, model_path, num_threads=os.cpu_count()):
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        for input_detail in input_details:
            input_name = input_detail["name"]
            input_index = input_detail["index"]

            if "bbox" in input_name:
                self.input_bbox_index = input_index
            elif "cov" in input_name:
                self.input_cov_index = input_index
            elif "max_output_size" in input_name:
                self.input_max_output_size_index = input_index
            elif "iou_threshold" in input_name:
                self.input_iou_threshold_index = input_index
            elif "score_threshold" in input_name:
                self.input_score_threshold_index = input_index

        for output_detail in output_details:
            output_shape = output_detail["shape"]
            output_index = output_detail["index"]

            if np.array_equal(output_shape, [1, 4]):
                self.output_bbox_index = output_index
            elif np.array_equal(output_shape, [1]):
                self.output_cov_index = output_index

    def predict(
        self, scores, boxes, max_output_size=20, iou_threshold=0.3, score_threshold=0.2
    ):
        self.interpreter.set_tensor(self.input_bbox_index, boxes)
        self.interpreter.set_tensor(self.input_cov_index, scores)
        self.interpreter.set_tensor(
            self.input_max_output_size_index,
            np.array([max_output_size], dtype=np.int32),
        )
        self.interpreter.set_tensor(
            self.input_iou_threshold_index, np.array([iou_threshold], dtype=np.float32)
        )
        self.interpreter.set_tensor(
            self.input_score_threshold_index,
            np.array([score_threshold], dtype=np.float32),
        )

        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_bbox_index)
        scores = self.interpreter.get_tensor(self.output_cov_index)

        return scores, boxes
