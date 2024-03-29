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


class DetectNetV2Model:
    def __init__(self, model_path, num_threads=os.cpu_count()):
        delegates = []

        for _ in range(2):
            try:
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads,
                    experimental_delegates=delegates,
                )

                self.interpreter.allocate_tensors()
            except RuntimeError as re:
                if len(
                    delegates
                ) == 0 and "Encountered unresolved custom op: ethos-u." in str(re):
                    # retry with the Ethos-U delegate
                    delegates = [tflite.load_delegate("/usr/lib/libethosu_delegate.so")]
                    continue

                raise re

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.output_0_index = output_details[0]["index"]
        self.output_1_index = output_details[1]["index"]

    def predict(self, x):
        x = (x - 128).astype(np.int8)
        x = np.expand_dims(x, axis=0)

        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()

        cov = self.interpreter.get_tensor(self.output_0_index)
        bbox = self.interpreter.get_tensor(self.output_1_index)

        return cov, bbox
