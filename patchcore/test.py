# # pretrained_custom_load='npz' in 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
# # print(pretrained_custom_load)

# # # from timm.models import create_model

# # # vit_model = create_model(
# # #     "vit_base_patch16_224",
# # #     pretrained=True,
# # #     num_classes=15,
# # #     drop_rate=0.0,
# # #     drop_path_rate=0.0,
# # #     drop_block_rate=None,
# # # )


# # import os
# # import sys
# # sys.path.append("/root/ljh/mucad")
# # import torch
# # import numpy as np
# # from text_prompt import Adapter, LearnablePrompt, TextEncoder
# # adapter = Adapter("cuda:0", 768, 768)
# # learnable_prompt = LearnablePrompt("cuda:0", 15, 768)
# # text_encoder = TextEncoder("cuda:0")
# # text = "normal object"
# # text_cls_feat = adapter(text_encoder(text, learnable_prompt()))
# import torch
# import open_clip
# import numpy as np
# import sys
# sys.path.append("/mnt/d/ljh/MUCAD/mucad")
# from text_prompt import Adapter
# from mucad_model import ConvAdapter, CrossAttention
# activations = {}
# def get_activation(name):
#     def hook(module, input, output):
#         activations[name] = output
#     return hook
# input = torch.randn(8, 3, 256, 256)
# model, _, preprocess = open_clip.create_model_and_transforms(
#     model_name="convnext_base_w", 
#     pretrained="/mnt/d/ljh/MUCAD/mucad/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin")
# getattr(model.visual.trunk.stages, "1").register_forward_hook(get_activation("1"))
# # input = preprocess(input)
# print(input.shape)
# model.eval()
# print(model)
# output = model.encode_image(input)
# print(output.shape)
# adapter = Adapter("cpu", 1024, 768)
# conv_adapter = ConvAdapter("cpu", 256, 768)
# # 打印捕获的激活值 # 1*32*32*256 -> 1*14*14*768
# for name, activation in activations.items():
#     print(f"Layer: {name}, Activation Shape: {activation.shape}")
#     conv_adapter_output = conv_adapter(activation)
#     print(conv_adapter_output.shape)
#     batch_size = conv_adapter_output.shape[0]
#     conv_feature = conv_adapter_output.permute(0, 2, 3, 1).reshape(batch_size, -1, 768) # 1*768*14*14 -> 1*14*14*768 -> 1*196*768
# vit_feature = torch.randn(8, 196, 768)
# cross_attn = CrossAttention(768, 12, 768, 768)
# cross_attn_output, _ = cross_attn(conv_feature, vit_feature, vit_feature)
# print(cross_attn_output.shape)
# for i, (name, module) in enumerate(model.visual.named_modules()):
#     print(name)
# import numpy as np
# import matplotlib.pyplot as plt
# data = np.array([0.1,0.3,0.8,0.1,0.5,10,12,14,17])
# # 指数变换
# exp_data = np.exp(data / 100) - 1  # 用100进行缩放以避免过大的数值

# # 绘制图形
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(data, marker='o', label='Original Data')
# plt.title('Original Data')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(exp_data, marker='o', label='Exponential Transformed Data', color='green')
# plt.title('Exponential Transformed Data')
# plt.grid(True)

# plt.show()

import numpy as np

def custom_sigmoid(x):
    k = 3  # 控制函数的陡峭程度
    b = 10  # 控制函数的中心位置
    return 1 / (1 + np.exp(-k * (x - b)))

# 测试函数
x_values = np.linspace(0, 20, 100)
y_values = custom_sigmoid(x_values)

import matplotlib.pyplot as plt
plt.plot(x_values, y_values)
plt.title('Custom Sigmoid Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()