import torch
from torch import nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=200, out_features=64),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, x):
        return self.model(x)


# 模型下载
# torch_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
torch_model = MyModel()
opt = torch.optim.Adam(torch_model.parameters(), lr=1e-5)
for ep in range(5):
    for _ in range(10):
        inputs = torch.randn(5, 5, 200)
        label = torch.randn(5, 5, 2)
        output = torch_model.forward(inputs)
        loss = torch.nn.CrossEntropyLoss()(output, label)
        opt.zero_grad()
        loss.backward()
        opt.step()


# 导出ONNX模型,这里的操作是为了转化成onnx格式的模型
# https://www.bilibili.com/video/BV1AU4y1t7Xi?p=31&vd_source=4555df39ac0c5857e7d031ed1f0f6189
dummy_input = torch.randn(1, 5, 200)
onnx_model_fp32 = 'output/model.onnx'
torch.onnx.export(torch_model, (dummy_input,), onnx_model_fp32, input_names=["test_input"], opset_version=11, verbose=False)

# 检查模型
model = onnx.load(onnx_model_fp32)
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))

# 模型量化
onnx_model_uint8 = 'output/model_quantize.onnx'
quantize_dynamic(model_input=Path(onnx_model_fp32), model_output=Path(onnx_model_uint8), weight_type=QuantType.QUInt8)

# 检查量化模型
model = onnx.load(onnx_model_uint8)
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))