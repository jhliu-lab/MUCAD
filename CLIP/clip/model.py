from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import sys
import torch.nn.functional as F
from torch import nn
from functools import partial
import sys
import cv2
sys.path.append("/home/ljh/MUCAD/mucad_v3")
print(sys.path)
from prompt import EPrompt
# from attention import PreT_Attention
from CLIP.clip.attention import PreT_Attention
from mucad_model import build_fusion_model, CrossAttention



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_prompt=False, use_text_prompt=False, use_vv_attention=False):
        super().__init__()
        if use_prompt or use_text_prompt:
            self.attn = PreT_Attention(d_model, n_head, use_vv_attention=use_vv_attention)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_prompt = use_prompt
        self.use_text_prompt = use_text_prompt
    
    def attention(self, x: torch.Tensor, prompt=None):
        if self.use_prompt or self.use_text_prompt:
            return self.attn(x, prompt)
        else:
            return self.attn(x, x, x)

    def forward(self, x: torch.Tensor, prompt=None):
        if self.use_prompt or self.use_text_prompt:
            x = x + self.attention(self.ln_1(x), prompt)
        else:
            x = x + self.attention(self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_feature_layer = None,
            img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', norm_layer=None, act_layer=None, 
            prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,
            use_g_prompt=False, g_prompt_length=None, g_prompt_layer_idx=None, use_prefix_tune_for_g_prompt=False,
            use_e_prompt=False, e_prompt_layer_idx=None, use_prefix_tune_for_e_prompt=False, same_key_value=False,
            prototype_size=5, feature_layer = [5], is_vision = True, use_text_prompt=False, use_vv_attention=False,
            text_prompt_layer_idx=[], text_wise_layer_prompt_len=1
        ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_e_prompt = use_e_prompt
        self.use_feature_layer = use_feature_layer
        self.e_prompt_layer_idx = e_prompt_layer_idx
        self.prompt_pool = prompt_pool
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.norm = nn.LayerNorm(normalized_shape=(197,768), eps=1e-6, dtype=torch.float16)
        self.class_token = class_token
        self.head_type = head_type
        self.global_pool = global_pool
        self.text_prompt_layer_idx = text_prompt_layer_idx
        self.is_vision = is_vision
        self.use_text_prompt = use_text_prompt
        self.text_wise_layer_prompt_len = text_wise_layer_prompt_len
        self.cross_attn = CrossAttention(768, 8, 768, 768)
        self.text_wise_layer_prompt = [torch.nn.Parameter(torch.randn((text_wise_layer_prompt_len, 512)))]*12
        ##################################################
        self.fusion_model = build_fusion_model('/home/ljh/MUCAD/mucad_v3/CLIP/ViT-B-16.pt')
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]) 
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        if use_e_prompt or use_text_prompt and e_prompt_layer_idx is not None:
            self.e_prompt = EPrompt(length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                    prompt_key_init=prompt_key_init, num_layers=num_e_prompt, use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                    num_heads=heads, same_key_value=same_key_value)
            self.cos_loss = torch.nn.CosineSimilarity()
            self.total_prompt_len = 0
            if self.prompt_pool:
                if not self.use_prefix_tune_for_e_prompt:
                    self.total_prompt_len += prompt_length * top_k * len(self.e_prompt_layer_idx)
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, 
                                                               heads, 
                                                               attn_mask, 
                                                               use_prompt=use_e_prompt, 
                                                               use_text_prompt=use_text_prompt,
                                                               use_vv_attention=use_vv_attention) for _ in range(layers)])
    
    def forward_features(self, 
                        x: torch.Tensor, task_id=-1, 
                        cls_features=None, train=False, 
                        query_task_id=None, use_prompt_for_wise_layer=False, 
                        vision_layer=5, cos_layers=[11]):
        if self.is_vision:
            if self.use_e_prompt:
                single_prompt_mask = torch.tensor(0).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                e_prompt_counter = -1
                res = self.e_prompt(x, prompt_mask=prompt_mask)
                e_prompt = res['batched_prompt']
                res['l_feat'] = []
                for i, block in enumerate(self.resblocks):
                        if i in self.e_prompt_layer_idx:
                            e_prompt_counter += 1
                            if self.use_prefix_tune_for_e_prompt:
                                # Prefix tunning, [B, 2, top_k * e_prompt_length, num_heads, embed_dim // num_heads]
                                x = block(x, prompt=e_prompt[e_prompt_counter])
                            else:
                                # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                                prompt = e_prompt[e_prompt_counter]
                                # print(torch.sum(prompt))
                                x = torch.cat([prompt, x], dim=1)
                                x = block(x)                    
                        else:
                            x = block(x)
                        if i in cos_layers:
                            res['l_feat'].append(x)
                        if(i==vision_layer):
                            if x.shape[0] == 12:
                                res['seg_feat'] = [x[:8,1:,:]]
                            else:
                                res['seg_feat'] = [x[:,1:,:]] 
            else:
                res = dict()
                res['l_feat'] = []
                for i, block in enumerate(self.resblocks):
                    x = block(x)
                    if(i==vision_layer):
                        # seg_feat shape = batch_size*196*768
                        if x.shape[0] == 12:
                            res['seg_feat'] = [x[:8,1:,:]]
                        else:
                            res['seg_feat'] = [x[:,1:,:]] 
                    if i in cos_layers:
                        res['l_feat'].append(x)
            x = self.norm(x)
            res['x'] = x
            res['g_feat'] = x
            return res
        else:
            # 文本编码器
            if self.use_text_prompt:
                x = x.permute(1,0,2) # BNC
                B, N, C = x.shape
                single_prompt_mask = torch.tensor(0).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                e_prompt_counter = -1
                res = self.e_prompt(x, prompt_mask=prompt_mask)
                e_prompt = res['batched_prompt']
                # res = dict()
                for i, block in enumerate(self.resblocks):
                        if i in self.text_prompt_layer_idx:
                            e_prompt_counter += 1
                            if self.use_prefix_tune_for_e_prompt:
                                if use_prompt_for_wise_layer:
                                    prompt = self.text_wise_layer_prompt[e_prompt_counter]
                                    x[:,N-self.text_wise_layer_prompt_len:N,:] = prompt
                                    x = block(x)
                                else :
                                    # Prefix tunning, [B, 2, top_k * e_prompt_length, num_heads, embed_dim // num_heads]
                                    x = block(x, prompt=e_prompt[e_prompt_counter])   
                            else:
                                # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                                prompt = self.text_wise_layer_prompt[e_prompt_counter]
                                # print(torch.sum(prompt))
                                x = torch.cat([prompt, x], dim=1)
                                x = block(x)                  
                        else:
                            x = block(x)
                x = x.permute(1,0,2) # NBC
            else:
                for block in self.resblocks:
                    x = block(x)
            return x
    
    def forward_head(self, res, pre_logits: bool = False, label=None, train=False, task_id=-1, image_path=None, text_feature=None):
        x = res['x']
        if x.shape[0] == 12:
            x = x[:8]
        # res['seg_feat'] = x[:,1:,:]
        if self.class_token and self.head_type == 'token':
            if self.prompt_pool:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier')
        # 这里的x就是最后一层vit提取到的特征 8*197*768
        res['pre_logits'] = x
        if(self.use_e_prompt and train):
            res['loss'] = torch.tensor(0).float().cuda()
            # labels shape = batch_size*196
            labels = torch.zeros((x.shape[0],14*14)).cuda()
            for i in range(x.shape[0]):
                if('mvtec2d' in image_path[i]):
                    # shape = 224*224*3
                    sam_score = cv2.imread(image_path[i].replace('mvtec2d','mvtec2d-sam-b'))
                elif('visa' in image_path[i]):
                    sam_score = cv2.imread(image_path[i].replace('visa','visa-sam-b'))
                    # 首先224*224*3 resize 得到 14*14*3 最后取灰度图像14*14展平为196 赋值给labels[i]
                elif('MPDD' in image_path[i]):
                     sam_score = cv2.imread(image_path[i].replace('MPDD','MPDD-sam-b'))
                elif('Retina_RESC_AD' in image_path[i]):
                    sam_score = cv2.imread(image_path[i].replace('Retina_RESC_AD', "Retina_RESC_AD-sam-b"))
                elif('btad' in image_path[i]):
                    sam_score = cv2.imread(image_path[i].replace('btad', "btad-sam-b"))
                labels[i] = torch.from_numpy(cv2.resize(sam_score,(14,14))[:,:,0].flatten()).cuda()
            res['loss'] = torch.tensor(0).float().cuda()
            # loss for sam seg_feat shape = batch_size*196*768
            for k in range(len(res['seg_feat'])):
                # 遍历batch_size中的图片特征，然后分别计算损失，最后将一个批次的损失相加
                res['loss'] += self.contrastive_loss(res['seg_feat'][k], labels, temperature=0.5)
        else:
            pass
        return res
    #cls_features=None, image_path=None, text_feature=None, conv_image_feat=None
    def forward(
        self, 
        x, 
        task_id=-1, 
        cls_features=None, 
        train=False, 
        pre_logits=False, 
        label=None, 
        image_path=None, 
        query_task_id=None, 
        text_feature=None, 
        conv_image_feat=None,
        use_prompt_for_wise_layer=False):
        if self.is_vision:
            res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train, query_task_id=query_task_id)
            res = self.forward_head(res, pre_logits, label, train, task_id=task_id, image_path=image_path, text_feature=text_feature)
            return res
        else:
            x = self.forward_features(x, use_prompt_for_wise_layer=use_prompt_for_wise_layer)
            return x

    def contrastive_loss(self, features, labels, temperature=0.5):
        # n, c, h, w = features.shape
        
        # 1. reshape feature
        # features = features.view(n, c, -1).permute(0, 2, 1)  # shape: [n, h*w, c]
        features_normalized = F.normalize(features, dim=2)

        # 2. calc similarity
        similarity_matrix = torch.bmm(features_normalized, features_normalized.transpose(1, 2)) / temperature

        # 3. make mask
        # labels = labels.view(n, -1)  # shape: [n, h*w]
        mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()

        # 4. get loss
        loss = (-similarity_matrix * mask + (1-mask) * similarity_matrix.exp()).mean()

        return loss

    def init_eprompt(self):
        self.e_prompt.__init__()

    def get_cur_prompt(self, dataloader_count=None):
        return self.e_prompt.get_prompt().clone().detach()
        # torch.save(self.e_prompt,'./models/prompt'+str(dataloader_count)+'.pt')

    def get_prompt(self):
        return self.e_prompt.get_prompt()
    
    def set_cur_prompt(self,saved_prompt, save_status=False, dataloader_count=None):
        self.e_prompt.set_prompt(saved_prompt, save_status=save_status)
        # self.e_prompt.load_state_dict(torch.load('./models/prompt'+str(dataloader_count)+'.pt'))

    def init_eprompt(self):
        self.e_prompt.__init__()

class VisionTransformer(nn.Module):
    def __init__(self, 
                 input_resolution: int, 
                 patch_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 output_dim: int, 
                 use_prompt:bool=False, 
                 use_vv_attention:bool=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width, layers, heads,
            num_classes=15, drop_rate=0.0,
            drop_path_rate=0.0,
            prompt_length=1,
            embedding_key='cls', prompt_init='uniform',
            prompt_pool=use_prompt, prompt_key=True,
            pool_size=1, top_k=1,batchwise_prompt=use_prompt,
            prompt_key_init='uniform', head_type='token',
            use_prompt_mask=use_prompt, use_prefix_tune_for_e_prompt=use_prompt, 
            use_e_prompt=use_prompt, e_prompt_layer_idx=[0,1,2,3,4,5,6,7,8,9,10,11],
            same_key_value=False, prototype_size=5, is_vision=True, use_vv_attention=use_vv_attention
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, task_id = -1, train=False, cls_features=None, image_path=None, text_feature=None, conv_image_feat=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND
        res = self.transformer(x, task_id=task_id, train=train, cls_features=cls_features, image_path=image_path, text_feature=text_feature, conv_image_feat=conv_image_feat)
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = res['x']
        x = self.ln_post(x[:, 0, :])
        res['cls_token_768'] = x
        if self.proj is not None:
            x = x @ self.proj
        res['cls_token_512'] = x
        return res

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 use_prompt=False,
                 use_text_prompt=False
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_prompt=use_prompt,
                use_vv_attention=False
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            is_vision=False,
            num_classes=15, drop_rate=0.0,
            drop_path_rate=0.0,
            prompt_length=1,
            embedding_key='cls', prompt_init='uniform',
            prompt_pool=use_text_prompt, prompt_key=use_text_prompt,
            pool_size=1, top_k=1,batchwise_prompt=use_text_prompt,
            prompt_key_init='uniform', head_type='token',
            use_prompt_mask=use_text_prompt, use_prefix_tune_for_e_prompt=use_text_prompt, 
            use_e_prompt=use_text_prompt, text_prompt_layer_idx=[],
            text_wise_layer_prompt_len = 1,
            same_key_value=False, prototype_size=5,
            use_text_prompt=use_text_prompt,
            embed_dim=512
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, train = False, task_id=-1, cls_features=None, image_path=None, text_feature=None, conv_image_feat=None):
        return self.visual(image.type(self.dtype), train = train,
                           task_id=task_id, cls_features=cls_features, 
                           image_path=image_path, text_feature=text_feature,
                           conv_image_feat=conv_image_feat)

    def encode_text(self, text, text_normal_feature_prompt=None, prompt_idxs=None, use_prompt_for_wise_layer=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if text_normal_feature_prompt is not None:
            # 加入可学习的正常文本提示
            text_normal_feature_prompt = text_normal_feature_prompt.to(dtype=torch.float16)
            assert len(prompt_idxs) == text_normal_feature_prompt.shape[0]
            x[:,prompt_idxs,:] = text_normal_feature_prompt
        x = x + self.positional_embedding.unsqueeze(0).to(dtype=self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_prompt_for_wise_layer=use_prompt_for_wise_layer)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # 3*77*512
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # 3*512
        # x = x[:,prompt_idxs,:] # 3*prompt_len*512
        x = torch.mean(x,dim=0) # 1*512
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, use_prompt=False, use_text_prompt=False):
    vit = "visual.proj" in state_dict
    
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, 
        transformer_layers, use_prompt=use_prompt, use_text_prompt=use_text_prompt
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print("missing_keys ------------------------------------------")
    # print(missing_keys)
    # print("expected_keys -----------------------------------------")
    # print(unexpected_keys)
    return model.eval()
