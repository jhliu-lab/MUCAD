"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.models
from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
import sys
sys.path.append("/home/ljh/MUCAD/mucad_v3")
sys.path.append("/home/ljh/MUCAD")
from mucad_model import  TextScore, get_similarity_map_
from CLIP.clip.clip import load, tokenize 
LOGGER = logging.getLogger(__name__)

class AdaptiveSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 1.5
        #self.k = torch.nn.Parameter(torch.tensor(1.0))
        #self.b = 8.0
        self.b = torch.nn.Parameter(torch.tensor(6.0))  # 偏移
        #self.b.requires_grad = True

    def forward(self, x):
        shifted_x = self.k * (x - self.b)  # 应用k和b的调整
        pos_mask = shifted_x >= 0
        neg_mask = shifted_x < 0
        y = torch.zeros_like(shifted_x)
        y[pos_mask] = 1.0 / (1 + torch.exp(-shifted_x[pos_mask]))
        y[neg_mask] = torch.exp(shifted_x[neg_mask]) / (torch.exp(shifted_x[neg_mask]) + 1)
        return y


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,    
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        layer_num=4,
        fb_args_k = 1.5,
        fb_args_b = 8.0,
        use_adapt_b_by_grad_down=False,
        **kwargs
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        # (k,b)
        self.adaption_sigmoid_list = {
            "mvtec" : [
                [AdaptiveSigmoid()]*15,
                [AdaptiveSigmoid()]*15
            ],
            "visa" : [
                [AdaptiveSigmoid()]*12,
                [AdaptiveSigmoid()]*12
            ]
        }
        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        self.vision_projection = [torch.nn.Linear(768, 768).cuda() for _ in range(layer_num)]
        
        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        # self.text_score = TextScore()

        self.featuresampler = featuresampler
        # self.fusion_mdoel = build_fusion_model('/home/ljh/MUCAD/mucad_v3/CLIP/ViT-B-16.pt')
        self.dataloader_count = 0
        self.model, _ = load("/home/ljh/MUCAD/mucad_v3/CLIP/ViT-B-16.pt")
        self.prompt_model, _ = load("/home/ljh/MUCAD/mucad_v3/CLIP/ViT-B-16.pt", use_prompt=True, use_text_prompt=False)
        self.model.to('cuda')
        self.prompt_model.to('cuda')
        for p in self.model.parameters():
            p.requires_grad = False
        for l in self.vision_projection:
            for p in l.parameters():
                p.requires_grad = False
        ## freeze args.freeze[blocks, patch_embed, cls_token] parameters
        ## print parameters
        for name, p in self.prompt_model.named_parameters():
            if "e_prompt" not in name:
                # 不是可训练参数，冻结
                p.requires_grad = False
            else:
                # print(name)
                p.requires_grad = True
    
    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def embed_text(self, texts, text_normal_feature_prompt=None, prompt_idxs=None, use_prompt_for_wise_layer=False):
        return self._embed_text(texts, text_normal_feature_prompt, prompt_idxs, use_prompt_for_wise_layer=use_prompt_for_wise_layer)
    
    def _embed_text(self, texts, text_normal_feature_prompt=None, prompt_idxs=None, use_prompt_for_wise_layer=False):
        text_tokens = tokenize(texts).to(self.device)
        text_features = self.prompt_model.encode_text(text=text_tokens, 
                                                      text_normal_feature_prompt=text_normal_feature_prompt, 
                                                      prompt_idxs=prompt_idxs, 
                                                      use_prompt_for_wise_layer=use_prompt_for_wise_layer)
        return text_features

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            # features = self.forward_modules["feature_aggregator"](images)
            features = self.model.encode_image(images)['seg_feat']
            for i in range(len(features)):
                features[i] = features[i].reshape(-1,14,14,768).permute(0,3,1,2)
    
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]


        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def embed_prompt(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed_prompt(input_image))
            return features
        return self._embed_prompt(data)

    def _embed_prompt(self, images, detach=True, provide_patch_shapes=False, text_feat=None, provide_res=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            # features = self.forward_modules["feature_aggregator"](images)
            # 获取训练数据的特征features shape = (bs, patch_num, dim)
            res = self.prompt_model.encode_image(images, text_feature=text_feat)
            features = res['seg_feat'] # [[8,196,768]]
            # features = self.prompt_model.encode_image(images, text_feature=text_feat)['seg_feat'] #[[8,196,768],...]
            for i in range(len(features)):
                # features[i] shape = (8,768, 14,14)
                features[i] = features[i].reshape(-1,14,14,768).permute(0,3,1,2)
        # features shape = (bs, patch_size*patch_size, c, patchsize, patchsize)
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        # patch_shapes shape = (14,14)
        patch_shapes = [x[1] for x in features]
        # features shape = (bs, 196, 768, 1, 1)
        features = [x[0] for x in features]
        """ 
        # text_feature and image_feature fusion
        bs, patch_num = features[0].shape()[:2]
        img_feature = features[0].reshape(bs, patch_num, -1)
        assert img_feature.shape()[-1] == 768, "img_feature shapre_error!"
        fusion_feature = self.fusion_mdoel(text_feature, img_feature)
        features = [fusion_feature.reshape(features[0].shape)]
        """
        # ref_num_patches shape = [14, 14]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        
        features = [x.reshape(-1, *x.shape[-3:]) for x in features] #(1568,768,1,1)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)#(1,1024)
        features = self.forward_modules["preadapt_aggregator"](features)#(1024,)
        # 脱离GPU，不保留梯度
        for key in res.keys():
            if key != 'reduce_sim' and isinstance(res[key], torch.Tensor):
                    res[key] = _detach(res[key])

        if provide_patch_shapes and provide_res:
            return _detach(features), patch_shapes, res
        elif provide_patch_shapes:
            return _detach(features), patch_shapes
        elif provide_res:
            return _detach(features), res
        return _detach(features)

    def _embed_train(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        # _ = self.model.eval()
        res = self.prompt_model.encode_image(images,task_id=self.dataloader_count, cls_features=None, train=True) # TODO: Train=True==neg
            # print(features.shape)
        return res

    def _embed_train_false(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        # _ = self.model.eval()
        res = self.prompt_model.encode_image(images,task_id=self.dataloader_count, cls_features=None, train=False) # TODO: Train=True==neg
            # print(features.shape)
        return res

    def _embed_train_sam(self, images, detach=True, provide_patch_shapes=False, image_path=None, text_cls_feat=None, conv_image_feat=None):
        """Returns feature embeddings for images."""
        res = self.prompt_model.encode_image(images,task_id=self.dataloader_count, cls_features=None, train=True, image_path=image_path, text_feature=text_cls_feat, conv_image_feat=conv_image_feat)
        return res

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        return features
    
    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        return features
    
    def fit_with_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        # 最远点采样
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        # self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def get_mem_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._get_mem_limit_size(training_data, limit_size)
        
    def _get_mem_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        return features
    
    def fit_with_limit_size_prompt(self, training_data, limit_size, text_feat=None):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_prompt(training_data, limit_size, text_feat)
        
    def _fill_memory_bank_with_limit_size_prompt(self, input_data, limit_size, text_feat=None):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed_prompt(input_image,text_feat=text_feat)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0) #  shape = (40964, 1024)
        features = self.featuresampler.run_with_limit_memory(features, limit_size) # shape = (1960,1024)
        self.anomaly_scorer.fit(detection_features=[features])
        return features

    def get_normal_prototypes(self, data, args):
        # switch to evaluation mode
        with torch.no_grad():
            cls_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model.encode_image(image)
                        cls_features = output['pre_logits']
                        cls_memory.append(cls_features.cpu())
        cls_prototypes = torch.cat([cls_f for cls_f in cls_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,random_state=0)
        labels = kmeans.fit_predict(cls_prototypes)
        representatives = torch.zeros(args.prototype_size,768)
        for i in range(args.prototype_size):
            cluster_tensors = cls_prototypes[labels==i]
            representative = np.mean(cluster_tensors,axis=0)
            representatives[i] = torch.from_numpy(representative)

        return representatives
    
    def get_normal_prototypes_instance(self, data, args):
        # switch to evaluation mode

        with torch.no_grad():
            cls_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model.encode_image(image)
                        cls_features = output['pre_logits']
                        cls_memory.append(cls_features.cpu())
        cls_prototypes = torch.cat([cls_f for cls_f in cls_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,n_init=10,max_iter=300).fit(cls_prototypes)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        representatives = torch.zeros(args.prototype_size,768)
        for i in range(args.prototype_size):
            representatives[i] = torch.from_numpy(centers[i])

        return representatives
    
    def get_normal_prototypes_seg(self, data, args):
        with torch.no_grad():
            seg_feat_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model.encode_image(image)
                        seg_feat = output['seg_feat']
                        seg_feat_memory.append(seg_feat[0].cpu())
        seg_prototypes = torch.cat([seg_feat.reshape(-1,196*4*768) for seg_feat in seg_feat_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,n_init=10,max_iter=300).fit(seg_prototypes)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        representatives = torch.zeros(args.prototype_size,196*4,768)
        for i in range(args.prototype_size):
            representatives[i] = torch.from_numpy(centers[i]).reshape(196*4,768)

        return representatives
    
    def get_normal_prototypes_seg_mean(self, data, args):
        with torch.no_grad():
            seg_feat_memory = list()
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()

                        output = self.model.encode_image(image)
                        seg_feat = output['seg_feat'][0]
                        seg_feat_memory.append(seg_feat.cpu())
        seg_prototypes = torch.cat([seg_feat.reshape(-1,196*4*768) for seg_feat in seg_feat_memory],dim=0).numpy()
        kmeans = KMeans(n_clusters=args.prototype_size,random_state=0)
        labels = kmeans.fit_predict(seg_prototypes)
        representatives = torch.zeros(args.prototype_size,196*4,768)
        for i in range(args.prototype_size):
            cluster_tensors = seg_prototypes[labels==i]
            representative = np.mean(cluster_tensors,axis=0)
            representatives[i] = torch.from_numpy(representative).reshape(196*4,768)

        return representatives

    def train_(self, data, dataloader_count, memory_feature):
        args = np.load('../args_dict.npy',allow_pickle=True).item()
        args.prototype_size = 5
        args.lr = 0.0005
        args.decay_epochs = 3#30
        args.warmup_epochs = 1#5
        args.cooldown_epochs = 1#10
        args.patience_epochs = 1#10
        optimizer = create_optimizer(args, self.prompt_model)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        # self.prompt_model.set_prompt_seg(dataloader_count,torch.from_numpy(memory_feature).cuda())
        prompt_cls_feature = self.get_normal_prototypes(data, args=args)
        self.prompt_model.set_prompt_cls(dataloader_count,prompt_cls_feature)
        prompt_seg_feature = self.get_normal_prototypes_seg(data, args=args)
        # prompt_seg_feature = self.get_normal_prototypes_seg_mean(data, args=args)
        self.prompt_model.set_prompt_seg(dataloader_count,prompt_seg_feature)
        # self.anomaly_scorer.fit(detection_features=[prompt_seg_feature.clone().detach().reshape(-1,768).cpu().numpy()])


        epochs = 10
        self.prompt_model.train()
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                    res = self._embed_train(image, provide_patch_shapes=True)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
                print("epoch:{} loss:{}".format(i,np.mean(loss_list)))    
            if lr_scheduler:
                lr_scheduler.step(i)
            # prompt_seg_feature = self.get_normal_prototypes_seg(data, args=args)
            # self.prompt_model.set_prompt_seg(dataloader_count,prompt_seg_feature)
        
        # prompt_seg_feature = prompt_seg_feature.reshape(-1,768)
        # print(prompt_seg_feature.shape)

        return prompt_seg_feature
    
    
    def train_contrastive(self, data, dataloader_count, memory_feature=None):
        args = np.load('../args_dict.npy',allow_pickle=True).item()
        args.prototype_size = 5
        args.lr = 0.0005
        # args.decay_epochs = 3#30
        # args.warmup_epochs = 1#5
        # args.cooldown_epochs = 1#10
        # args.patience_epochs = 1#10
        args.decay_epochs = 10#30
        args.warmup_epochs = 2#5
        args.cooldown_epochs = 3#10
        args.patience_epochs = 3#10
        optimizer = create_optimizer(args, self.prompt_model)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        epochs = 20
        self.prompt_model.train()
        self.prompt_model.train_contrastive = True
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    if(image["image"].shape[0]<2):
                        continue
                    if isinstance(image, dict):
                        image = image["image"].cuda()
                    res = self._embed_train_false(image, provide_patch_shapes=True)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
                print("epoch:{} loss:{}".format(i,np.mean(loss_list)))    
            if lr_scheduler:
                lr_scheduler.step(i)
        
        # prompt_seg_feature = prompt_seg_feature.reshape(-1,768)
        # print(prompt_seg_feature.shape)

        # return prompt_seg_feature

    #aug data1,data2 contrastive
    # def train_con(self, data, dataloader_count, memory_feature):
        
    def train_sam(self, data, dataloader_count, memory_feature=None):
        args = np.load('../args_dict.npy',allow_pickle=True).item()
        args.lr = 0.0005
        args.decay_epochs = 3#30
        args.warmup_epochs = 1#5
        args.cooldown_epochs = 1#10
        args.patience_epochs = 1#10
        # args.decay_epochs = 10#30
        # args.warmup_epochs = 2#5
        # args.cooldown_epochs = 3#10
        # args.patience_epochs = 3#10
        optimizer = create_optimizer(args, self.prompt_model)
        self.dataloader_count = dataloader_count

        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        epochs = 10
        self.prompt_model.train()
        self.prompt_model.train_contrastive = True
        for i in range(epochs):
            loss_list = []
            with tqdm.tqdm(data, desc="training...", leave=False) as data_iterator:
                for image in data_iterator:
                    # if(image["image"].shape[0]<2):
                    #     continue
                    if isinstance(image, dict):
                        image_paths = image["image_path"]
                        image = image["image"].cuda()
                    # res = self._embed_train_false(image, provide_patch_shapes=True)
                    res = self._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths)
                    loss = res['loss']
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    if(loss!=0):
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), args.clip_grad)
                    optimizer.step()
                print("epoch:{} loss:{}".format(i,np.mean(loss_list)))    
            if lr_scheduler:
                lr_scheduler.step(i)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)#TODO: baseline
            # features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    
    def predict_prompt(self, data, text_feature=None, multi_layer = 1, 
                       dataset_name="mvtec", query_task_id=0, is_basic = False,
                       use_adapt_b_by_grad_down=False, is_predict=False):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_prompt(data, 
                                                   text_feature, multi_layer, 
                                                   dataset_name=dataset_name, 
                                                   query_task_id=query_task_id, is_basic=is_basic,
                                                   use_adapt_b_by_grad_down=use_adapt_b_by_grad_down,
                                                   is_predict=is_predict)
        return self._predict_prompt(data, text_feature, multi_layer, 
                                    dataset_name=dataset_name, query_task_id=query_task_id, 
                                    is_basic=is_basic,use_adapt_b_by_grad_down=use_adapt_b_by_grad_down,
                                    is_predict=is_predict)

    def _predict_dataloader_prompt(self, dataloader, text_feature=None, multi_layer = 1, 
                                   dataset_name="mvtec", query_task_id=0, is_basic=False,
                                   use_adapt_b_by_grad_down=False, is_predict=False):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            total_loss = 0
            batch_num = 0
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist()) # 8*1
                    masks_gt.extend(image["mask"].numpy().tolist()) # 8*1*224*224
                    image = image["image"] # 8*3*224*224
                _scores, _masks, loss = self._predict_prompt(image, text_feature, multi_layer, dataset_name=dataset_name, 
                                                       query_task_id=query_task_id, is_basic=is_basic,
                                                       use_adapt_b_by_grad_down=use_adapt_b_by_grad_down,
                                                       is_predict=is_predict) # _scores=>shape[8,], _masks=>shape(8*[224,224])
                if loss is not None:
                    total_loss += loss
                    batch_num += 1
                for score, mask in zip(_scores, _masks): # 图像级别预测分数以及像素级别mask（224*224）
                    scores.append(score)
                    masks.append(mask)
            if total_loss != 0:
                total_loss = total_loss / batch_num
        return scores, masks, labels_gt, masks_gt, total_loss

    def normalize_images(self, anomaly_maps, k=2, b=10):
        """
        对每个图像进行归一化, 使其像素值在0到1之间。
        
        :param images: 形状为 (batch_size, height, width) 的图像数组
        :return: 归一化后的图像数组
        """
        anomaly_maps = np.array(anomaly_maps)
        anomaly_maps = self.custom_sigmoid(anomaly_maps, k=k, b=b)
        return anomaly_maps
    
    def custom_activation(self, x, k=2, m = 0.8, n=0.0):
        # x = torch.from_numpy(x)
        x = torch.where(x > m, x * k, x)
        x = torch.where(x > 0.5, x / k, x)
        x = torch.where(x < n, x / k, x)
        return x
    
    def get_op_sigmoid_loss(self, x):
        # 损失1：让正常样本的归一化值接近0
        loss_mean = torch.mean(x)
        
        # 损失2：稀疏性约束（抑制高值）
        loss_sparse = torch.mean(torch.clamp(x - 0.1, min=0))
        
        # 损失3：熵最大化（让低值更集中）
        p = x
        loss_entropy = -torch.mean(p * torch.log(p + 1e-6) + (1 - p) * torch.log(1 - p + 1e-6))
        
        # 总损失
        loss = loss_mean + 0.5 * loss_sparse + 0.1 * loss_entropy
        return loss


    def custom_sigmoid(self, x, k=2, b=10):
        """
        自定义的Sigmoid函数 加入分正负的计算逻辑（向量化版本）。
        参数：
        - x: 输入的numpy数组
        - k: 控制函数的陡峭程度
        - b: 控制函数的中心位置
        """
        shifted_x = k * (x - b)  # 应用k和b的调整
        pos_mask = shifted_x >= 0
        neg_mask = shifted_x < 0

        y = np.zeros_like(shifted_x)
        y[pos_mask] = 1.0 / (1 + np.exp(-shifted_x[pos_mask]))
        y[neg_mask] = np.exp(shifted_x[neg_mask]) / (np.exp(shifted_x[neg_mask]) + 1)
        return y
    
    def _predict_prompt(self, images, text_feature=None, multi_layer = 1, dataset_name="mvtec", 
                        query_task_id=0, is_basic=False,use_adapt_b_by_grad_down=False,is_predict=False):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0] #8
        if use_adapt_b_by_grad_down:
            with torch.no_grad():
                # features=>(1024,) patch_shape=(14*14)
                features, patch_shapes, res = self._embed_prompt(images, provide_patch_shapes=True, provide_res=True)#TODO: baseline
                # features, patch_shapes = self._embed(images, provide_patch_shapes=True)
                features = np.asarray(features)
                # print(features.shape) #(1024,)
                # features = np.repeat(features,2,axis=1)
                # 1024的特征向量与196/1960*1024的特征向量中进行KNN搜索得到分数(得到逐patch的分数)
                patch_scores = image_scores = self.anomaly_scorer.predict([features])[0] # (1568,)
                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                ) # 8*196
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1) # 8*196*1
                image_scores = self.patch_maker.score(image_scores) # 8
                patch_scores = self.patch_maker.unpatch_scores(
                    patch_scores, batchsize=batchsize
                ) #8*196
                if text_feature is not None:  
                    image_feat = np.array(res['cls_token_768'])
                    image_feat = torch.tensor(image_feat).detach().cuda() #8*197*768
                    multi_layer_image_feature = res['l_feat']
                    sim_map = None
                    for i in range(multi_layer):
                        # img_feat = multi_layer_image_feature[len(multi_layer_image_feature) - i - 1]
                        img_feat = multi_layer_image_feature[i]
                        img_feat = img_feat.detach().cuda()
                        img_feat = img_feat.to(dtype=torch.float32)
                        if sim_map is None:
                            # sim_map = get_similarity_map_(self.vision_projection[i](img_feat), text_feature)
                            sim_map = get_similarity_map_(img_feat, text_feature)
                        else:
                            # sim_map += get_similarity_map_(self.vision_projection[i](img_feat), text_feature)
                            sim_map += get_similarity_map_(img_feat, text_feature)
                    sim_map = torch.sigmoid(sim_map)
                    # images_scores_text = torch.max(input=sim_map, dim=0) 
                    # images_scores_text = torch.tensor([torch.max(map_) for map_ in sim_map])
                    bs = image_feat.shape[0]
                    sim_map = sim_map.reshape(bs, 224, 224).cpu()
                    sim_map = torch.stack([torch.from_numpy(gaussian_filter(score.numpy(), sigma=10)) for score in sim_map])
                    wights = sim_map
                scales = patch_shapes[0] # 14*14
                patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])#8*14*14
            ######################################################################
            loss1 = 0
            loss2 = 0
            # 归一化后再与文本的融合
            task_type = 'mvtec'
            if "visa" in dataset_name:
                task_type = 'visa'
            if use_adapt_b_by_grad_down:
                if text_feature is not None:
                    if is_basic:
                        #wights_norm = self.adaption_sigmoid_list_basic[task_type][0][query_task_id](wights*10)
                        pass
                    else:
                        wights = torch.tensor(wights)
                        wights_norm = self.adaption_sigmoid_list[task_type][0][query_task_id](wights*10)
                        loss1 = self.get_op_sigmoid_loss(wights_norm)
                        wights_fusion = wights_norm.detach().numpy()
                if is_basic:
                    pass
                else:
                    patch_scores = torch.tensor(patch_scores)
                    masks_norm = self.adaption_sigmoid_list[task_type][1][query_task_id](patch_scores)
                    loss2 = self.get_op_sigmoid_loss(masks_norm)
                    masks_norm = self.anomaly_segmentor.convert_to_segmentation(masks_norm)# 8*224*224 np.array
                    masks_fusion = np.array(masks_norm)
                # ######################################################################
                if text_feature is not None:
                    # beta = 0.5
                    alpha = 0.9
                    masks_fusion = np.asarray(masks_norm) * alpha + np.asarray(wights_fusion)*(1-alpha)
                    masks_fusion = np.asarray(torch.stack([torch.from_numpy(gaussian_filter(mask, sigma=5)) for mask in masks_fusion]))
                return [score for score in image_scores], [mask for mask in masks_fusion], loss1 + loss2
            else:
                masks= self.anomaly_segmentor.convert_to_segmentation(patch_scores)# 8*224*224 np.array
                masks = np.asarray(masks)
                if text_feature is not None:
                    alpha = 0.9
                    masks_fusion = np.asarray(masks) * alpha + np.asarray(wights*10)*(1-alpha)
                    masks_fusion = np.asarray(torch.stack([torch.from_numpy(gaussian_filter(mask, sigma=5)) for mask in masks_fusion]))
                return [score for score in image_scores], [mask for mask in masks_fusion], None
        else:
            with torch.no_grad():
                # features=>(1024,) patch_shape=(14*14)
                features, patch_shapes, res = self._embed_prompt(images, provide_patch_shapes=True, provide_res=True)#TODO: baseline
                # features, patch_shapes = self._embed(images, provide_patch_shapes=True)
                features = np.asarray(features)
                # print(features.shape) #(1024,)
                # features = np.repeat(features,2,axis=1)
                # 1024的特征向量与196/1960*1024的特征向量中进行KNN搜索得到分数(得到逐patch的分数)
                patch_scores = image_scores = self.anomaly_scorer.predict([features])[0] # (1568,)
                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                ) # 8*196
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1) # 8*196*1
                image_scores = self.patch_maker.score(image_scores) # 8
                patch_scores = self.patch_maker.unpatch_scores(
                    patch_scores, batchsize=batchsize
                ) #8*196
                if text_feature is not None:  
                    image_feat = np.array(res['cls_token_768'])
                    image_feat = torch.tensor(image_feat).detach().cuda() #8*197*768
                    multi_layer_image_feature = res['l_feat']
                    sim_map = None
                    for i in range(multi_layer):
                        # img_feat = multi_layer_image_feature[len(multi_layer_image_feature) - i - 1]
                        img_feat = multi_layer_image_feature[i]
                        img_feat = img_feat.detach().cuda()
                        img_feat = img_feat.to(dtype=torch.float32)
                        if sim_map is None:
                            # sim_map = get_similarity_map_(self.vision_projection[i](img_feat), text_feature)
                            sim_map = get_similarity_map_(img_feat, text_feature)
                        else:
                            # sim_map += get_similarity_map_(self.vision_projection[i](img_feat), text_feature)
                            sim_map += get_similarity_map_(img_feat, text_feature)
                    sim_map = torch.sigmoid(sim_map)
                    # images_scores_text = torch.max(input=sim_map, dim=0) 
                    # images_scores_text = torch.tensor([torch.max(map_) for map_ in sim_map])
                    bs = image_feat.shape[0]
                    sim_map = sim_map.reshape(bs, 224, 224).cpu()
                    sim_map = torch.stack([torch.from_numpy(gaussian_filter(score.numpy(), sigma=10)) for score in sim_map])
                    wights = sim_map
                scales = patch_shapes[0] # 14*14
                patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])#8*14*14
                ######################################################################
                loss1 = 0
                loss2 = 0
                # 归一化后再与文本的融合
                task_type = 'mvtec'
                if "visa" in dataset_name:
                    task_type = 'visa'
                if use_adapt_b_by_grad_down:
                    if text_feature is not None:
                        if is_basic:
                            #wights_norm = self.adaption_sigmoid_list_basic[task_type][0][query_task_id](wights*10)
                            pass
                        else:
                            wights = torch.tensor(wights)
                            wights_norm = self.adaption_sigmoid_list[task_type][0][query_task_id](wights*10)
                            loss1 = self.get_op_sigmoid_loss(wights_norm)
                            wights_fusion = np.asarray(wights_norm)
                    if is_basic:
                        pass
                    else:
                        patch_scores = torch.tensor(patch_scores)
                        masks_norm = self.adaption_sigmoid_list[task_type][1][query_task_id](patch_scores)
                        loss2 = self.get_op_sigmoid_loss(masks_norm)
                        masks_norm = self.anomaly_segmentor.convert_to_segmentation(masks_norm)# 8*224*224 np.array
                        masks_fusion = np.asarray(masks_norm)
                    # ######################################################################
                    if text_feature is not None:
                        # beta = 0.5
                        alpha = 0.9
                        masks_fusion = np.asarray(masks_norm) * alpha + np.asarray(wights_fusion)*(1-alpha)
                        masks_fusion = np.asarray(torch.stack([torch.from_numpy(gaussian_filter(mask, sigma=5)) for mask in masks_fusion]))
                    return [score for score in image_scores], [mask for mask in masks_fusion], loss1 + loss2
                else:
                    masks= self.anomaly_segmentor.convert_to_segmentation(patch_scores)# 8*224*224 np.array
                    masks = np.asarray(masks)
                    if text_feature is not None:
                        alpha = 0.9
                        masks_fusion = np.asarray(masks) * alpha + np.asarray(wights*10)*(1-alpha)
                        masks_fusion = np.asarray(torch.stack([torch.from_numpy(gaussian_filter(mask, sigma=5)) for mask in masks_fusion]))
                    return [score for score in image_scores], [mask for mask in masks_fusion], None

    def _predict_past_tasks(self, features, data):
        pass
            
    def _fit_past_tasks(self, features, data):
        pass
        

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
