#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
#

import sys
import time

import numpy as np

from detect_net_v2_model import DetectNetV2Model

NUM_RUNS = 100

for model_path in sys.argv[1:]:
    for num_threads in range(1, 3):
        model = DetectNetV2Model(model_path, num_threads=num_threads)

        # dummy
        model.predict(np.zeros((544, 960, 3), dtype=np.uint8))

        total_time = 0

        for _ in range(NUM_RUNS):
            x = np.random.randint(0, 255, (544, 960, 3), np.uint8)

            predict_start = time.time()
            model.predict(x)
            predict_end = time.time()

            total_time += predict_end - predict_start

        print(
            f"{model_path} (num_threads = {num_threads}): Average inference latency = {total_time * 1000 / NUM_RUNS} ms"
        )
