import torch
import sys
from scipy.ndimage import gaussian_filter
import torch.nn.modules
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import torch.nn.functional as F
from text_prompt import contrastive_sem, Adapter, compute_similarity, get_similarity_map

class MUCAD(torch.nn.Module):
    def __init__(self, 
                 device, 
                 patchcore, 
                 learnable_text_prompt=None, 
                 adapter=None, 
                 use_vision_init_prompt=False, 
                 layer_num=4,
                **kwargs):
        super(MUCAD, self).__init__()
        self.device = device
        self.PatchCore = patchcore
        self.learnable_text_prompt=learnable_text_prompt
        self.adapter = adapter
        self.layer_num = layer_num
        if use_vision_init_prompt:
            vision_prompt_dim = kwargs['vision_prompt_dim']
            self.prompt_linear = torch.nn.Linear(vision_prompt_dim, layer_num*512)
        self.dice_loss = BinaryDiceLoss()
        # self.project_linear = torch.nn.Linear(768, 768).cuda()
        # self.fusion_model = MultiModalFusion(768, 8).cuda()
        # self.vision_proj = torch.nn.Parameter((768, 768))

    def init_prompt_with_vision_feature(self, vision_feature): # 196*1024 --> 4*2*3*8*64
        vision_feature = torch.flatten(torch.tensor(vision_feature))
        vision_feature = vision_feature.to(dtype=self.learnable_text_prompt.prompts.dtype)
        vision_init_prompt = self.prompt_linear(vision_feature).reshape(self.layer_num, 512)
        # self.PatchCore.prompt_model.transformer.text_wise_layer_prompt = torch.nn.Parameter(vision_init_prompt)
        self.learnable_text_prompt.prompts = torch.nn.Parameter(vision_init_prompt)
         

    def forward(self, texts, prompt_idxs = None, use_adapter=False, use_prompt_for_wise_layer=False,image_feat=None):
        # 文本特征提取
        if prompt_idxs is not None:
            text_feature = self.PatchCore.embed_text(texts, 
                                                     self.learnable_text_prompt(), 
                                                     prompt_idxs=prompt_idxs,
                                                     use_prompt_for_wise_layer=use_prompt_for_wise_layer)
        else:
            text_feature = self.PatchCore.embed_text(texts)
        if use_adapter and self.adapter is not None:
            text_feature = self.adapter(text_feature) # 1*output_dim
        if image_feat is not None:
            src_shape = text_feature.shape
            image_feat = image_feat.detach().cuda()
            image_feat = image_feat.to(dtype=torch.float32)            
            image_feat = self.project_linear(image_feat)
            B, _, _ = image_feat.shape
            text_feature = torch.stack([text_feature.reshape(1, -1)]*B)
            text_feature = self.fusion_model(text_feature.reshape(B, 1, -1), image_feat)
            text_feature = torch.mean(text_feature, dim=0).reshape(src_shape)
        return text_feature

    def get_sem_loss(self, image_feat, text_feature):
        return contrastive_sem(image_feat, text_feat=text_feature)
    
    def get_similarity_map(self, image_features, text_features, image_size=224):
        similarity, _ = compute_similarity(image_features, text_features)
        return 1 - get_similarity_map(similarity, image_size).permute(0, 3, 1, 2) # batch_size*1*224*224

    def get_dice_loss(self, input, targets):
        return self.dice_loss(input, targets)
    
    def get_L2_loss(self, input, targets):
        return torch.mean((input - targets) ** 2)
    
    def get_pixle_loss(self, sim_map, gt=None): # sim_map shape = (12*224*224)
        # gt = torch.zeros_like(sim_map)
        # flag = torch.randn_like(gt)
        # flag = torch.where(flag > 0.5, 0, 0.1)
        # gt = flag
        # sim_map = torch.sigmoid(sim_map)
        gt = gt.cuda()
        # pixle_loss = sim_map.mean() # 效果较dice_loss更好
        pixle_loss = F.mse_loss(sim_map, gt)
        # pixle_loss = self.get_L2_loss(sim_map, gt)
        # pixle_loss = sim_map.mean()
        # pixle_loss = F.binary_cross_entropy(sim_map, gt, reduction="mean")
        # pixle_loss = self.combined_loss(sim_map, gt)
        # if sim_map.shape[0] > 8:
        #     pixle_loss = F.mse_loss(sim_map, gt) + sim_map[:8].mean() - sim_map[8:].mean()
        # else :
        #     pixle_loss = F.mse_loss(sim_map, gt) + sim_map.mean()
        # pixle_loss = self.get_dice_loss(sim_map, gt) + F.mse_loss(sim_map, gt)
        # dice_loss 的对比消融
        return pixle_loss
    
    # 可视化特征图
    def visualize_feature_maps(self, feature_maps, num_cols=4):

        feature_maps = feature_maps[:,1:,:].cpu()
        B, C, L = feature_maps[:,1:,:].shape
        feature_maps = feature_maps.reshape(B, 14, 14, 3, 16, 16).permute(0, 1, 4, 2, 5, 3).reshape(B, 224, 224, 3)
        num_features = feature_maps.shape[0]
        num_rows = (num_features + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
        axes = axes.flatten()

        for i in range(num_features):
            if i < num_features:
                ax = axes[i]
                # 选择一个通道来可视化，这里选择第一个通道
                feature_map = feature_maps[i, :, :, 0].detach().numpy()
                ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Feature Map {i+1}')
                ax.axis('off')
            else:
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()
   
    # 归一化特征图到 0 到 1 之间
    def normalize_feature_maps(self, feature_maps):
        min_val = feature_maps.min()
        max_val = feature_maps.max()
        normalized_feature_maps = (feature_maps - min_val) / (max_val - min_val)
        return normalized_feature_maps

    # 保存特征图为图片文件
    def save_feature_maps(self, feature_maps, output_dir=None, num_cols=4):
        feature_maps = self.normalize_feature_maps(feature_maps)
        feature_maps = feature_maps[:,1:,:].cpu()
        B, C, L = feature_maps[:,1:,:].shape
        feature_maps = feature_maps.reshape(B, 14, 14, 3, 16, 16).permute(0, 1, 4, 2, 5, 3).reshape(B, 224, 224, 3)
        output_dir = '/home/ljh/MUCAD/mucad_v3/vision'
        os.makedirs(output_dir, exist_ok=True)

        for i in range(B):
        
            # 选择一个通道来可视化，这里选择第一个通道
            feature_map = feature_maps[i, :, :, :].detach().numpy()       
            # 保存单个特征图为图片文件
            save_path = os.path.join(output_dir, f'feature_map_{i+1}.png')
            plt.imsave(save_path, feature_map, cmap='viridis')

    def get_image_loss(self, p, label): # 余弦相似度损失
        # # 类型转换和设备转移
        # image_features = image_features.to(dtype=text_features.dtype)
        # if image_features.device != text_features.device:
        #     image_features = image_features.to(device=text_features.device)
        # # 归一化特征
        # image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
        # # 计算相似度矩阵
        # cos_sim = image_features_norm @ text_features_norm.t() # 8*1
        # logits = torch.sigmoid(1 - cos_sim) #
        image_loss = F.binary_cross_entropy(p.squeeze().cuda(), label.cuda()) 
        # image_loss = F.cross_entropy(p.squeeze().cuda(), label.cuda())
        return image_loss, None
    
    def add_random_noise(self, image, noise_type='gaussian', noise_level=0.01, noise_region_ratio=0.1, noise_mask=None, device='cpu'):
        """
        在图像中添加随机噪声，并生成噪声掩码。
        :param image: 输入图像，形状为 (C, H, W) 或 (H, W)
        :param noise_type: 噪声类型，可选值为 'gaussian', 'salt_and_pepper', 'uniform', 'clustered'
        :param noise_level: 噪声强度，具体含义取决于噪声类型
        :param noise_region_ratio: 噪声区域占整个图像中心的比例（对于 'clustered' 类型，此参数不起作用）
        :param noise_mask: 如果传入了噪声掩码，可以复用这个掩码，否则自动生成
        :param device: 设备类型，可选值为 'cpu' 或 'cuda'
        :return: 带噪声的图像和噪声掩码
        """
        if isinstance(image, torch.Tensor):
            image_np = image.to(device=device).numpy()
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            raise ValueError("图像必须是 numpy 数组或 torch 张量")

        # 确保图像为 float 类型
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)
        flag = False
        # 确保图像在 [0, 1] 范围内
        if image_np.max() > 1:
            image_np /= 255.0
            flag = True

        # 获取图像的形状
        if image_np.ndim == 2:
            image_np = image_np[np.newaxis, ...]  # 添加通道维度
        C, H, W = image_np.shape
        ratio_center = [0.01, 0.25, 0.5, 0.75, 0.9]
        ratio_idx = np.random.randint(0,4)
        # # 定义中心区域
        # center_x, center_y = W // 2, H // 2
        center_width = int(W * ratio_center[ratio_idx])  # 中心区域宽度
        center_height = int(H * ratio_center[ratio_idx])  # 中心区域高度
        # center_mask = np.zeros((H, W), dtype=np.float32)
        # center_mask[center_y - center_height // 2:center_y + center_height // 2, 
        #             center_x - center_width // 2:center_x + center_width // 2] = 1.0
        # 定义中心区域
        ratio_idx = np.random.randint(0,3)
        center_x, center_y = W // 2, H // 2
        # radius = int(min(W, H) * ratio_center[ratio_idx])  # 圆形区域的半径
        radius = int(min(W, H) * 0.5)
        center_mask = np.zeros((H, W), dtype=np.float32)
        y, x = np.ogrid[:H, :W]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        center_mask[mask] = 1.0

        # 计算中心区域的像素数量
        center_pixels = np.sum(center_mask)

        # 生成噪声掩码
        if noise_mask is None:
            noise_mask = np.zeros((H, W), dtype=np.float32)
            if noise_type != 'clustered':
                num_noisy_pixels = int(center_pixels * noise_region_ratio)
                # 获取中心区域内的所有像素索引
                center_indices = np.argwhere(center_mask == 1)
                # 选择噪声像素的索引
                selected_indices = np.random.choice(len(center_indices), num_noisy_pixels, replace=False)
                # 更新噪声掩码
                noise_mask[center_indices[selected_indices][:, 0], center_indices[selected_indices][:, 1]] = 1.0
            else:
                # 'clustered' 类型的噪声
                num_clusters = np.random.randint(10, 20)  # 1到5个聚集区
                for _ in range(num_clusters):
                    cluster_center = (np.random.randint(center_y - center_height // 2, center_y + center_height // 2),
                                    np.random.randint(center_x - center_width // 2, center_x + center_width // 2))
                    a = np.random.randint(1, 100)  # 半长轴
                    b = np.random.randint(1, 100)  # 半短轴
                    angle = np.random.uniform(0, 2 * np.pi)  # 旋转角度
                    y, x = np.ogrid[:H, :W]
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    x_rotated = x - cluster_center[1]
                    y_rotated = y - cluster_center[0]
                    ellipse_mask = ((x_rotated * cos_angle + y_rotated * sin_angle) ** 2 / a ** 2 +
                                    (x_rotated * sin_angle - y_rotated * cos_angle) ** 2 / b ** 2) <= 1
                    mask = (ellipse_mask * center_mask) > 0  # 确保聚集区在中心区域内
                    noise_mask[mask] = 1.0
        # 生成噪声
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, (C, H, W))
            # 仅保留噪声掩码中为1的区域
            noise = noise * noise_mask[None, :, :]
        elif noise_type == 'salt_and_pepper':
            noise = np.zeros((C, H, W))
            num_salt = np.ceil(center_pixels * noise_region_ratio / 2)
            num_pepper = np.ceil(center_pixels * noise_region_ratio / 2)
            # 获取中心区域内的所有像素索引
            center_indices = np.argwhere(center_mask == 1)
            # 选择盐噪声像素的索引
            selected_salt_indices = np.random.choice(len(center_indices), int(num_salt), replace=False)
            # 选择胡椒噪声像素的索引
            selected_pepper_indices = np.random.choice(len(center_indices), int(num_pepper), replace=False)
            # 更新噪声掩码
            noise_mask[center_indices[selected_salt_indices][:, 0], center_indices[selected_salt_indices][:, 1]] = 1
            noise_mask[center_indices[selected_pepper_indices][:, 0], center_indices[selected_pepper_indices][:, 1]] = 1
            # 设置噪声值
            noise[:, center_indices[selected_salt_indices][:, 0], center_indices[selected_salt_indices][:, 1]] = 1
            noise[:, center_indices[selected_pepper_indices][:, 0], center_indices[selected_pepper_indices][:, 1]] = -1
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, (C, H, W))
            # 仅保留噪声掩码中为1的区域
            noise = noise * noise_mask[None, :, :]
        elif noise_type == 'clustered':
            noise = np.random.normal(0, noise_level, (C, H, W)) * noise_mask[None, :, :]
        else:
            raise ValueError("不支持的噪声类型")

        # 添加噪声
        noisy_image = image_np + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        if flag:
            noisy_image *= 255
        # 如果输入图像是 torch 张量，返回 torch 张量
        if isinstance(image, torch.Tensor):
            noisy_image = torch.from_numpy(noisy_image).to(device)
            noise_mask = torch.from_numpy(noise_mask).to(device)

        return noisy_image, noise_mask.reshape(1, H, W)
    
def get_similarity_map_(image_features, text_features, image_size=224):
        similarity, _ = compute_similarity(image_features, text_features)
        # 1 - similarity 为异常的概率, similarity 为正常的概率
        return 1 - get_similarity_map(similarity, image_size).permute(0, 3, 1, 2)

def get_text_probs(image_features, text_features):
        # 类型转换和设备转移
        image_features = image_features.to(dtype=text_features.dtype)
        if image_features.device != text_features.device:
            image_features = image_features.to(device=text_features.device)
        # 归一化特征
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
        # 计算相似度矩阵
        cos_sim = image_features_norm @ text_features_norm.t() # 8*1
        cos_sim = 1-cos_sim # 8*1, 异常的概率
        return cos_sim.flatten().cpu()

class CrossAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, kdim, vdim):
        super(CrossAttention, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, kdim=kdim, vdim=vdim)
    
    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_output_weights


class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(SelfAttention, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        x = x.to(dtype=torch.float16)
        attn_output, attn_output_weights = self.multihead_attn(x, x, x)
        return attn_output, attn_output_weights


class SAFModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SAFModel, self).__init__()
        self.cross_attention= CrossAttention(embed_dim, num_heads, embed_dim, embed_dim)
        
    def forward(self, image_feat, text_feat):
        # text_feat = torch.stack([text_feat[0]]*196) # shape = 196*768
        # text_feat = torch.stack([text_feat]*image_feat.shape[0])
        return self.cross_attention(text_feat, image_feat, image_feat)[0]


class ResidualSAFModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_saf=2):
        super(ResidualSAFModel, self).__init__()
        self.attns = torch.nn.ModuleList([SelfAttention(embed_dim, num_heads)]*num_saf)
        self.saf_layers = torch.nn.ModuleList([SAFModel(embed_dim, num_heads)]*num_saf)
        self.num_saf = num_saf
    def forward(self, image_feat, text_feat):
        x = image_feat
        for i in range(self.num_saf):
            x = self.saf_layers[i](self.attns[i](x)[0], text_feat)
        return image_feat + x


# 自注意线性融合模块
class SelfAttentionLinearFusionModule(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_saf=2, num_res_saf=3) -> None:
        super(SelfAttentionLinearFusionModule, self).__init__()
        self.residual_saf_layers = torch.nn.ModuleList([ResidualSAFModel(embed_dim, num_heads, num_saf)]*num_res_saf)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        # self.norm = torch.nn.LayerNorm()

    def forward(self, image_feat, text_feat):
        for block in self.residual_saf_layers:
            image_feat = block(image_feat, text_feat)
        return image_feat + self.linear(self.attn(image_feat)[0])


class TextScore(torch.nn.Module):
    def __init__(self):
        super(TextScore, self).__init__()


    def forward(self, image_features, text_feature, need_pixel=False):
        """
        image_features shape [8,196,768]
        text_feature shape [1,768]
        """
        bs = image_features.shape[0]
        # 1. 特征预处理
        # 将文本特征扩展到与图像特征相同的维度
        text_feature_expanded = text_feature.unsqueeze(0).expand(bs, 196, 768)
        image_features = image_features.cuda()
        # 2. 计算特征相似度
        cosine_similarity = F.cosine_similarity(text_feature_expanded, image_features, dim=2)
        # cosine_similarity 形状为 (8, 196)
        anomaly_scores = F.sigmoid(cosine_similarity)
        # 3. 获取异常分数
        # anomaly_scores = 1 - anomaly_scores # 取相似度的负值作为异常分数
        # anomaly_scores 形状为 (8, 196)
        if not need_pixel:
            return anomaly_scores
        # 4. 上采样
        # 将196个patch的异常分数上采样到224x224
        anomaly_scores = anomaly_scores.view(bs, 1, 14, 14)  # 重塑为 (8, 1, 14, 14)
        anomaly_scores = F.interpolate(anomaly_scores, size=(224, 224), mode='bilinear', align_corners=False)
        anomaly_scores = anomaly_scores.squeeze()  # 形状为 (8, 224, 224)
        anomaly_scores = anomaly_scores.cpu()
        # 5. 高斯平滑
        anomaly_scores = torch.stack([torch.from_numpy(gaussian_filter(score.numpy(), sigma=1.0)) for score in anomaly_scores])

        # 最终的异常分割图
        anomaly_maps = anomaly_scores

        # print(anomaly_maps.shape)  # 输出 (8, 224, 224)
        return anomaly_maps

class MultiModalFusion(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiModalFusion, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 多头注意力机制的层
        self.multihead_attn = torch.nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)

        # 线性层和激活函数用于融合后的特征
        self.fc = torch.nn.Linear(embed_size, embed_size)
        self.relu = torch.nn.ReLU()

    def forward(self, text_features, visual_features):
        """
        :param text_features: 文本模态特征, 形状 (seq_len_text, batch_size, embed_size)
        :param visual_features: 视觉模态特征, 形状 (seq_len_visual, batch_size, embed_size)
        :param text_mask: 文本模态的掩码, 形状 (batch_size, seq_len_text)
        :param visual_mask: 视觉模态的掩码, 形状 (batch_size, seq_len_visual)
        :return: 融合后的特征
        """
        # # 将文本和视觉模态对齐成相同长度（通过填充较短的模态）
        # seq_len_text = text_features.size(0)
        # seq_len_visual = visual_features.size(0)

        # if seq_len_text < seq_len_visual:
        #     padding = visual_features[:seq_len_text]
        #     text_features = torch.cat([text_features, padding], dim=0)
        #     if text_mask is not None:
        #         text_mask = torch.cat([text_mask, text_mask.new_zeros((text_mask.size(0), seq_len_visual - seq_len_text))], dim=1)
        # elif seq_len_visual < seq_len_text:
        #     padding = text_features[:seq_len_visual]
        #     visual_features = torch.cat([visual_features, padding], dim=0)
        #     if visual_mask is not None:
        #         visual_mask = torch.cat([visual_mask, visual_mask.new_zeros((visual_mask.size(0), seq_len_text - seq_len_visual))], dim=1)

        # # 将文本和视觉特征输入到多头注意力中
        # # 使用掩码来忽略填充部分
        # if text_mask is not None and visual_mask is not None:
        #     combined_mask = torch.cat([text_mask, visual_mask], dim=1)  # 组合两个模态的掩码
        # else:
        #     combined_mask = None

        # 进行多头注意力计算
        fused_features, _ = self.multihead_attn(text_features, visual_features, visual_features)

        # 使用全连接层进行进一步处理
        fused_features = self.fc(fused_features)
        fused_features = self.relu(fused_features)

        return fused_features


def convert_weights(model: torch.nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, torch.nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", 'proj']:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def convert_state_dict(state_dict):
    state_dict_ = dict()
    k = 5
    for i in range(3):
        for j in range(2):
            state_dict_['residual_saf_layers.'+str(i)+'.attns.'+str(j)+".multihead_attn.in_proj_weight"] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.in_proj_weight']
            state_dict_['residual_saf_layers.'+str(i)+'.attns.'+str(j)+".multihead_attn.in_proj_bias"] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.in_proj_bias']
            state_dict_['residual_saf_layers.'+str(i)+'.attns.'+str(j)+".multihead_attn.out_proj.weight"] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.out_proj.weight']
            state_dict_['residual_saf_layers.'+str(i)+'.attns.'+str(j)+".multihead_attn.out_proj.bias"] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.out_proj.bias']
            k += 1
    k = 11
    state_dict_['attn.multihead_attn.in_proj_weight'] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.in_proj_weight']
    state_dict_['attn.multihead_attn.in_proj_bias'] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.in_proj_bias']
    state_dict_['attn.multihead_attn.out_proj.weight'] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.out_proj.weight']
    state_dict_['attns.multihead_attn.out_proj.bias'] = state_dict['visual.transformer.resblocks.'+str(k)+'.attn.out_proj.bias']
    return state_dict_

def build_fusion_model(path=None):
    self_attn_linear_fusion = SelfAttentionLinearFusionModule(768, 8)
    if path is not None:
        model = torch.jit.load(path)
        state_dict_clip = convert_state_dict(model.state_dict())
        convert_weights(self_attn_linear_fusion)
        a, b = self_attn_linear_fusion.load_state_dict(state_dict_clip, strict=False)
        # print(a)
        # print(b)
    # 冻结自注意力层
    for name, p in self_attn_linear_fusion.named_parameters():
        if name in state_dict_clip:
            p.requires_grad = False
    return self_attn_linear_fusion

class BinaryDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        input = input.cuda()
        targets = targets.cuda()
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss

def sigmoid(x, gain=1.0, offset=0.5):
    """
    Sigmoid function to stretch values.
    
    Parameters:
    - x: Input array or value.
    - gain: Controls the steepness of the curve.
    - offset: Controls the midpoint of the curve.
    
    Returns:
    - Output array or value after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-gain * (x - offset)))

def post_image(seg_map, gain=5, offset=0):
    for i, map in enumerate(seg_map):
        flag = np.any(map > 1)
        if flag:
            max_value = np.max(map)
            map = map / max_value # 确保在0到1之间
        map = sigmoid(map, gain=gain, offset=offset)
        if flag:
            map = map * max_value
        seg_map[i] = map
    return seg_map
# model = build_fusion_model("/home/ljh/MUCAD/mucad_v3_visa/CLIP/ViT-B-16.pt")

def validation_loss(x, y):
    loss = torch.nn.BCELoss()(x, y)
    return loss