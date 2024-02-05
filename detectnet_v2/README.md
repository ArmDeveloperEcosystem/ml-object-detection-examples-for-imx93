# ML Object Detection DetectNet V2 example for i.MX 93

Example application for using [NVIDIA TAO DetectNet V2 models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_detectnet_v2) on the [NXP i.MX 93 SoC](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors/i-mx-9-processors/i-mx-93-applications-processor-family-arm-cortex-a55-ml-acceleration-power-efficient-mpu:i.MX93).

* [model_conversion.ipynb](model_conversion.ipynb): Jupyter Notebook to convert ONNX model to TensorFlow Lite model

* [`create_nms_model.py`](create_nms_model.py): Python script to create [`nms_model.tflite`](nms_model.tflite)

* Python Inference Application
   * [`main.py`](main.py): Inference Application entry
   * [`detect_net_v2_model.py`](detect_net_v2_model): Class to wrap DetectNet V2 TensorFlow Lite model
   * [`nms_model.py`](nms_model.py): Class to wrap DetectNet V2 TensorFlow Lite model
