import torch
import torch.nn as nn

torch.manual_seed(42)

# 定义输入数据
input_data = torch.randn(2, 4)  # 2个样本，4个特征
print('Input Data:', input_data)
'''Input Data: tensor([[ 0.3367,  0.1288,  0.2345,  0.2303],
        [-1.1229, -0.1863,  2.2082, -0.6380]])'''

# nn.Linear: 全连接层/线性层
linear_layer = nn.Linear(4, 3)  # 输入特征维度为4，输出特征维度为3
print('Linear Layer:',linear_layer.weight.data, linear_layer.bias.data)
'''Linear Layer: tensor([[ 0.3854,  0.0739, -0.2334,  0.1274],
        [-0.2304, -0.0586, -0.2031,  0.3317],
        [-0.3947, -0.2305, -0.1412, -0.3006]]) tensor([ 0.0472, -0.4938,  0.4516])'''

output = linear_layer(input_data)
print("Linear Layer Output:")
print(output)
print(input_data@linear_layer.weight.t()+linear_layer.bias)
print()
'''Linear Layer Output:
tensor([[ 0.1611, -0.5502,  0.1866],
        [-0.9961, -0.8843,  0.8177]], grad_fn=<AddmmBackward0>)
tensor([[ 0.1611, -0.5502,  0.1866],
        [-0.9961, -0.8843,  0.8177]], grad_fn=<AddBackward0>)'''