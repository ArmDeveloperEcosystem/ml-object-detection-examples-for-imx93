#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
#

import os
import sys
import time

import cv2
import numpy as np

from detect_net_v2_model import DetectNetV2Model
from nms_model import NMSModel
from post_processing_model import PostProcessingModel

model_path = sys.argv[1]

os.environ["QT_QPA_PLATFORM"] = ""
os.environ["DISPLAY"] = ":0"

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

max_output_sizes = [20, 20, 20]

iou_thresholds = [0.5, 0.5, 0.5]

score_thresholds = [0.4, 0.2, 0.2]

min_height = 20

# v4l2-ctl -d /dev/video0 --list-formats-ext
vid = cv2.VideoCapture(
    "v4l2src ! video/x-raw,width=1280,height=720 ! autovideoconvert ! "
    "videoscale ! video/x-raw,width=960,height=540 ! appsink drop=true sync=false",
    cv2.CAP_GSTREAMER,
)

model = DetectNetV2Model(model_path)
post_processor = PostProcessingModel((34, 60))
nms_model = NMSModel("nms_model.tflite")

while True:
    capture_start = time.time()
    ret, frame = vid.read()
    capture_end = time.time()
    print("capture", (capture_end - capture_start) * 1000)

    if not ret:
        break

    pad_start = time.time()
    frame = np.pad(frame, pad_width=((0, 4), (0, 0), (0, 0)))
    pad_end = time.time()
    print("pad", (pad_end - pad_start) * 1000)

    convert_start = time.time()
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convert_end = time.time()
    print("convert", (convert_end - convert_start) * 1000)

    predict_start = time.time()
    cov, bbox = model.predict(x)
    predict_end = time.time()
    print("predict", (predict_end - predict_start) * 1000)

    post_process_start = time.time()
    cov, bbox = post_processor.predict(cov, bbox)
    post_process_end = time.time()
    print("post process", (post_process_end - post_process_start) * 1000)

    for label in range(cov.shape[-1]):
        nms_start = time.time()
        nms_scores, nms_boxes = nms_model.predict(
            cov[0, :, label],
            bbox[0, :, label, :],
            max_output_sizes[label],
            iou_thresholds[label],
            score_thresholds[label],
        )
        nms_end = time.time()
        print("nms", (nms_end - nms_start) * 1000)

        for j in range(nms_scores.shape[0]):
            if nms_scores[j] == 0:
                break

            x1, y1, x2, y2 = nms_boxes[j, :].astype(np.int32)

            height = y2 - y1

            if height < min_height:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colors[label],
                2,
            )

    cv2.imshow(f"DetectNet V2: {model_path}", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
