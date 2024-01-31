#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
#

import numpy as np


class PostProcessingModel:
    def __init__(self, input_shape, stride=16, scale=35.0, offset=0.5):
        self.stride = stride
        self.scale = scale
        self.offset = offset

        self.centers_x = np.arange(input_shape[1]) * stride + offset
        self.centers_y = np.arange(input_shape[0]) * stride + offset

        self.centers_x = self.centers_x[:, np.newaxis]
        self.centers_y = self.centers_y[:, np.newaxis, np.newaxis]

    def predict(self, cov, bbox):
        bbox = np.reshape(
            bbox, [cov.shape[0], cov.shape[1], cov.shape[2], cov.shape[3], 4]
        )

        bbox[:, :, :, :, 0] = self.centers_x - self.scale * bbox[:, :, :, :, 0]
        bbox[:, :, :, :, 1] = self.centers_y - self.scale * bbox[:, :, :, :, 1]
        bbox[:, :, :, :, 2] = self.centers_x + self.scale * bbox[:, :, :, :, 2]
        bbox[:, :, :, :, 3] = self.centers_y + self.scale * bbox[:, :, :, :, 3]

        cov = np.reshape(cov, [cov.shape[0], cov.shape[1] * cov.shape[2], cov.shape[3]])
        bbox = np.reshape(
            bbox,
            [
                bbox.shape[0],
                bbox.shape[1] * bbox.shape[2],
                bbox.shape[3],
                bbox.shape[4],
            ],
        )

        return cov, bbox
