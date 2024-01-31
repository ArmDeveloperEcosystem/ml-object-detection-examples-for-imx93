#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
#

import tensorflow as tf

num_boxes = 34 * 60

cov = tf.keras.Input((), batch_size=num_boxes, name="cov")
bbox = tf.keras.Input((4), batch_size=num_boxes, name="bbox")
max_output_size = tf.keras.Input(
    (), batch_size=1, name="max_output_size", dtype=tf.int32
)
iou_threshold = tf.keras.Input((), batch_size=1, name="iou_threshold")
score_threshold = tf.keras.Input((), batch_size=1, name="score_threshold")

print("cov", cov.shape, cov.dtype)
print("bbox", bbox.shape, bbox.dtype)
print("max_output_size", max_output_size.shape, max_output_size.dtype)
print("iou_threshold", iou_threshold.shape, iou_threshold.dtype)
print("score_threshold", score_threshold.shape, score_threshold.dtype)

selected_indices, scores = tf.image.non_max_suppression_with_scores(
    boxes=bbox,
    scores=cov,
    max_output_size=max_output_size[0],
    iou_threshold=iou_threshold[0],
    score_threshold=score_threshold[0],
)

print("scores", scores.shape, scores.dtype)

boxes = tf.gather(bbox, selected_indices, name="boxes")

print("boxes", boxes.shape, boxes.dtype)

nms_model = tf.keras.Model(
    inputs=[cov, bbox, max_output_size, iou_threshold, score_threshold],
    outputs={"scores": scores, "boxes": boxes},
)

nms_model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(nms_model)
tflite_model = converter.convert()

with open("nms_model.tflite", "wb") as f:
    f.write(tflite_model)
