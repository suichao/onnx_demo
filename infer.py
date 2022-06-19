import onnxruntime as ort
import torch
import numpy as np


# ONNX推理函数
def inference(model_name, in_data):
    ort_session = ort.InferenceSession(model_name)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(
        None,
        {input_name: in_data},
    )

    return outputs


data = np.random.random((1, 5, 200)).astype('float32')
res = inference("output/model.onnx", data)
print(res)