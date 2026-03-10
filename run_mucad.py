import contextlib
import logging
import os
import sys
sys.path.append("/home/ljh/MUCAD/mucad_v3")
import click
import numpy as np
import torch
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
import warnings
import random
from metric_utils import find_optimal_threshold
import cv2
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
###################################################
from mucad_model import MUCAD
from text_prompt import LearnablePrompt, Adapter, get_texts
###################################################
import argparse
LOGGER = logging.getLogger(__name__)

_DATASETS = {
	"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
	"visa": ["patchcore.datasets.visa", "VisADatatset"]	
}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)                                                                                                                                                                                                                                                                                                                
@click.option("--log_group", type=str, default="group")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
@click.option("--log_project", type=str, default="project")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--memory_size", type=int, default=196, show_default=True)
@click.option("--epochs_num", type=int, default=50, show_default=True)
@click.option("--key_size", type=int, default=196, show_default=True)
@click.option("--basic_size", type=int, default=1960, show_default=True)
def main(**kwargs):
	pass


@main.result_callback()
def run(
	methods,
	results_path,
	gpu,
	seed,
	log_group,
	log_project,
	save_segmentation_images,
	save_patchcore_model,
	memory_size,
	epochs_num,
	key_size,
	basic_size,
):
	methods = {key: item for (key, item) in methods}
	warnings.filterwarnings("ignore", category=UserWarning)
	run_save_path = patchcore.utils.create_storage_folder(
		results_path, log_project, log_group, mode="iterate"
	)
	run_save_path_nolimit = patchcore.utils.create_storage_folder(
		results_path+'_nolimit', log_project, log_group, mode="iterate"
	)

	list_of_dataloaders = methods["get_dataloaders"](seed)

	device = patchcore.utils.set_torch_device(gpu)
	# Device context here is specifically set and used later
	# because there was GPU memory-bleeding which I could only fix with
	# context managers.
	device_context = (
		torch.cuda.device("cuda:{}".format(device.index))
		if "cuda" in device.type.lower()
		else contextlib.suppress()
	)
	result_collect = []
	result_collect_nolimit = []
	#key memory, prompt feature
	key_feature_list = [0]*15
	memory_feature_list = [0]*15
	prompt_list = [0]*15
	text_prompt_list = [0]*15
	text_prompt_len = 5
	multi_layer = 1
	use_adapt_b_by_grad_down = False
	only_vision = False
	aggregator_list = [0]*15
	use_vision_init_prompt = False
	mucad_model = MUCAD(
		device, 
		None, 
		None, 
		Adapter(device, 512, 768),
		use_vision_init_prompt=use_vision_init_prompt,
		vision_prompt_dim = 196*1024,
		layer_num=text_prompt_len
	)
	# 冻结部分参数权重
	for p in mucad_model.adapter.parameters():
		p.requires_grad = False
	for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
		LOGGER.info(
			"training dataset [{}] ({}/{})...".format(
				dataloaders["training"].name,
				dataloader_count + 1,
				len(list_of_dataloaders),
			)
		)
		# 当前子数据集
		patchcore.utils.fix_seeds(seed, device) 
		dataset_name = dataloaders["training"].name
		with device_context:
			torch.cuda.empty_cache()
			imagesize = dataloaders["training"].dataset.imagesize
			sampler = methods["get_sampler"](
				device,
			)
			PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
			if len(PatchCore_list) > 1:
				LOGGER.info(
					"Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
				)
			##########################################################################################################
			if not only_vision:
				mucad_model.learnable_text_prompt= LearnablePrompt(device, text_prompt_len, 512)
			mucad_model.PatchCore = PatchCore_list[0]
			######################################
			torch.cuda.empty_cache()
			if mucad_model.PatchCore.backbone.seed is not None:
				patchcore.utils.fix_seeds(mucad_model.PatchCore.backbone.seed, device)
			LOGGER.info(
				"Training models ({}/{})".format(1, len(PatchCore_list))
			)
			torch.cuda.empty_cache()
			# current_task_id task_num
			mucad_model.PatchCore.set_dataloadercount(dataloader_count)
			key_feature = mucad_model.PatchCore.fit_with_limit_size(dataloaders["training"], key_size)
			key_feature_list[dataloader_count] = key_feature
			if use_vision_init_prompt:
				# 使用视觉特征初始化文本提示
				mucad_model.init_prompt_with_vision_feature(key_feature)
			use_basic = False
			start_time = time.time()
			pr_auroc = 0
			basic_pr_auroc = 0
			basic_pr_ap = 0
			pr_ap = 0
			pr_k_i = 0
			basic_pr_k_i = 0
			args = np.load('./args_dict.npy',allow_pickle=True).item()
			args.lr = 0.00005
			args.decay_epochs = 15#30 15
			args.warmup_epochs = 3#5 30 
			args.cooldown_epochs = 5#10 10 
			args.patience_epochs = 5#10 10
			use_sam = True
			# 文本分支参数
			# 适配器
			if not only_vision:
				sem_paramter_list = list()
				# 可学习提示
				for p in mucad_model.learnable_text_prompt.parameters():
					sem_paramter_list.append(p)
				if not use_sam:
					# 去除SAM视觉提示用于学习正常样本特征    
					for p in mucad_model.PatchCore.prompt_model.visual.transformer.e_prompt.parameters():
						sem_paramter_list.append(p)     
				optimizer = create_optimizer(args, sem_paramter_list)
			if use_sam:
				# 视觉分支参数
				sam_paramter_list = []
				for p in mucad_model.PatchCore.prompt_model.visual.transformer.e_prompt.parameters():
					sam_paramter_list.append(p)
				sam_args = args
				sam_args.lr = 0.00005
				optimizer_sam =  create_optimizer(sam_args, sam_paramter_list)
			epochs = epochs_num
			if not only_vision:
				mucad_model.learnable_text_prompt.train()
			mucad_model.PatchCore.prompt_model.train()
			# mucad_model.PatchCore.prompt_model.train_contrastive = True
			if args.sched != 'constant':
				if not only_vision:
					lr_scheduler, _ = create_scheduler(args, optimizer)
				if use_sam:
					lr_scheduler_sam, _ = create_scheduler(sam_args, optimizer_sam)
			elif args.sched == 'constant':
				if not only_vision:
					lr_scheduler = None
				lr_scheduler_sam = None
			best_auroc,best_full_pixel_auroc,best_img_ap,best_pixel_ap,best_pixel_pro,best_time_cost = 0,0,0,0,0,0
			best_basic_auroc,best_basic_full_pixel_auroc,best_basic_img_ap,best_basic_pixel_ap,best_basic_pixel_pro,best_basic_time_cost = 0,0,0,0,0,0
			# train process
			for epoch in range(epochs):
				for i, PatchCore in enumerate([mucad_model.PatchCore]):
					torch.cuda.empty_cache()
					# '''
					if not only_vision:
						mucad_model.learnable_text_prompt.train()
					PatchCore.prompt_model.train()
					loss_list = []
					sam_loss_list = []
					image_loss_list = []
					patch_loss_list = []
					with tqdm.tqdm(dataloaders["training"], desc="training...", leave=False) as data_iterator:
						for image in data_iterator:
							if(image["image"].shape[0]<2): 
								continue
							if isinstance(image, dict):
								image_paths = image["image_path"]
								image = image["image"].cuda()# tensor类型
								B, C, H, W = image.shape
								gt = torch.zeros((B, 1, H, W)) #  
								target = torch.zeros((B,)) # 
								if B == 8:
									noise_image_num = 4
									# 生成噪音图像
									gt = torch.zeros((B+noise_image_num, 1, H, W)) #  
									target = torch.zeros((B+noise_image_num,)) # 
									images = torch.zeros((B+noise_image_num, C, H, W))
									images[:B, :, :, :] = image
									for i in range(noise_image_num):
										# noise_level = torch.randint(1, 30, (1,)) / 100.0
										# noise_ratio = torch.randint(1, 50, (1,)) / 100.0
										noise_image, noise_mask = mucad_model.add_random_noise(
																			image[i],
																			noise_level=1,
																			noise_region_ratio=1,
																			noise_type="gaussian")
										images[B+i] = noise_image
										gt[B+i] = noise_mask
										target[B+i] = 1
									image = images.cuda()

							# 得到图像特征和结构损失
							res = PatchCore._embed_train_sam(image, provide_patch_shapes=True, image_path=image_paths, text_cls_feat=None)
							if not only_vision:
								multi_layer_image_feature = res['l_feat'] # list 
								texts, prompt_idxs = get_texts(prompt_len=text_prompt_len, dataset_count=dataloader_count, only_text=False)
								text_features = mucad_model(texts, prompt_idxs= prompt_idxs, use_adapter=True, use_prompt_for_wise_layer=True, image_feat=None) # 1*768
								patch_loss = None
								sim_map = None
								for i in range(multi_layer):
									img_feat = multi_layer_image_feature[i]
									if use_sam:
										img_feat = img_feat.detach().cuda()
									img_feat = img_feat.to(dtype=torch.float32)
									if sim_map is None:							
										sim_map = mucad_model.get_similarity_map(img_feat, text_features)
									else:									
										sim_map += mucad_model.get_similarity_map(img_feat, text_features)
								sim_map = torch.sigmoid(sim_map)
								patch_loss = mucad_model.get_pixle_loss(sim_map, gt)
								loss = patch_loss  #+ image_loss
								patch_loss_list.append(patch_loss.item())	
								optimizer.zero_grad()
								loss_list.append(loss.item())
								if(loss!=0):
									loss.backward()
							if use_sam:
								optimizer_sam.zero_grad()
								sam_loss = res['loss']
								sam_loss_list.append(sam_loss.item())
								if(sam_loss!=0):
									sam_loss.backward()
								sam_loss_list.append(sam_loss.item())
							torch.nn.utils.clip_grad_norm_(mucad_model.PatchCore.prompt_model.parameters(), args.clip_grad)
							if not only_vision:
								optimizer.step()
							if use_sam:
								optimizer_sam.step()
						if not only_vision:
							if use_sam:
								print("epoch:{}  patch_loss:{} sam_loss:{} ".format(
								epoch, np.mean(patch_loss_list), np.mean(sam_loss_list)))
							else:
								print("epoch:{} patch_loss:{}".format(epoch, np.mean(patch_loss_list)))
							if lr_scheduler:
								lr_scheduler.step(i)  
						else:
							print("epoch:{} sam_loss:{} ".format(
								epoch, np.mean(sam_loss_list))) 
					if use_sam:
						if lr_scheduler_sam:
							lr_scheduler_sam.step(i)
					PatchCore.prompt_model.eval()
					mucad_model.eval()
					if not only_vision:
						# 获取文本
						texts, prompt_idxs = get_texts(prompt_len=text_prompt_len, dataset_count=dataloader_count, only_text=False)
						# 得到文本特征
						text_feat = mucad_model(texts, prompt_idxs=prompt_idxs, use_adapter=True, use_prompt_for_wise_layer=True) # 1*768
					basic_scores = None
					basic_segmentations = None
					if use_basic:
						nolimimit_memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], basic_size)
						# # 检索正常知识用于进行 异常分数计算
						PatchCore.anomaly_scorer.fit(detection_features=[nolimimit_memory_feature])
						basic_scores, basic_segmentations, basic_labels_gt, basic_masks_gt, _ = PatchCore.predict_prompt(
							dataloaders["testing"], text_feat if not only_vision else None, multi_layer, dataset_name=dataset_name, query_task_id=dataloader_count,
							is_basic=True
						)	
					basic_end_time = time.time()
					start_time_fps = time.time()
					# # 将memory大小限制在196，限制知识大小
					memory_feature = PatchCore.fit_with_limit_size_prompt(dataloaders["training"], memory_size)
					# 正常知识放入memory bank中用于检索
					PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
					scores, segmentations, labels_gt, masks_gt, _= PatchCore.predict_prompt(
						dataloaders["testing"], text_feat if not only_vision else None,
						multi_layer, dataset_name=dataset_name, query_task_id=dataloader_count, 
						use_adapt_b_by_grad_down=use_adapt_b_by_grad_down
					)
					end_time_fps = time.time()
					fps = (end_time_fps - start_time_fps)/len(dataloaders["testing"])
					print(fps)
				if dataset_name.startswith("mvtec"):
					anomaly_labels = [
						x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
					]
				elif dataset_name.startswith("visa"):
					anomaly_labels = [
						x[1] != "normal" for x in dataloaders["testing"].dataset.data_to_iterate
					]
				if dataset_name.startswith("mvtec"):
					basic_anomaly_labels = [
						x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
					]
				elif dataset_name.startswith("visa"):
					basic_anomaly_labels = [
						x[1] != "normal" for x in dataloaders["testing"].dataset.data_to_iterate
					]
				def calculation_eval(scores, segmentations, basic_scores=None, basic_segmentations=None):
					aggregator = {"scores": [], "segmentations": []}
					# aggregator 
					aggregator["scores"].append(scores)
					aggregator["segmentations"].append(segmentations)
					end_time = time.time()	
					# 评估
					scores = np.array(aggregator["scores"])
					min_scores = scores.min(axis=-1).reshape(-1, 1)
					max_scores = scores.max(axis=-1).reshape(-1, 1)
					scores = (scores - min_scores) / (max_scores - min_scores)
					scores = np.mean(scores, axis=0)
					segmentations = np.array(aggregator["segmentations"])
					min_scores = (
						segmentations.reshape(len(segmentations), -1)
						.min(axis=-1)
						.reshape(-1, 1, 1, 1)
					)
					max_scores = (
						segmentations.reshape(len(segmentations), -1)
						.max(axis=-1)
						.reshape(-1, 1, 1, 1)
					)
					segmentations = (segmentations - min_scores) / (max_scores - min_scores)
					segmentations = np.mean(segmentations, axis=0)
					# src_scores1 = scores
					# src_segmentations1 = segmentations
					time_cost = (end_time - basic_end_time)/len(dataloaders["testing"])
					ap_seg = np.asarray(segmentations)
					ap_seg = ap_seg.flatten()
					# LOGGER.info("Computing evaluation metrics.")
					auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
						scores, anomaly_labels
					)["auroc"]
					ap_mask = np.asarray(masks_gt)
					ap_mask = ap_mask.flatten().astype(np.int32)
					pixel_ap = average_precision_score(ap_mask,ap_seg)
					# metric without limit
					if use_basic:
						basic_aggregator = {"scores": [], "segmentations": []}
						basic_aggregator["scores"].append(basic_scores)
						basic_aggregator["segmentations"].append(basic_segmentations)
						basic_scores = np.array(basic_aggregator["scores"])
						basic_min_scores = basic_scores.min(axis=-1).reshape(-1, 1)
						basic_max_scores = basic_scores.max(axis=-1).reshape(-1, 1)
						basic_scores = (basic_scores - basic_min_scores) / (basic_max_scores - basic_min_scores)
						basic_scores = np.mean(basic_scores, axis=0)
						basic_segmentations = np.array(basic_aggregator["segmentations"])
						basic_min_scores = (
							basic_segmentations.reshape(len(basic_segmentations), -1)
							.min(axis=-1)
							.reshape(-1, 1, 1, 1)
						)
						basic_max_scores = (
							basic_segmentations.reshape(len(basic_segmentations), -1)
							.max(axis=-1)
							.reshape(-1, 1, 1, 1)
						)
						basic_segmentations = (basic_segmentations - basic_min_scores) / (basic_max_scores - basic_min_scores)
						basic_segmentations = np.mean(basic_segmentations, axis=0)
						
						basic_time_cost = (basic_end_time - start_time)/len(dataloaders["testing"])
						
						basic_ap_seg = np.asarray(basic_segmentations)
						basic_ap_seg = basic_ap_seg.flatten()
						basic_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
							basic_scores, basic_anomaly_labels
						)["auroc"]
						basic_ap_mask = np.asarray(basic_masks_gt)
						basic_ap_mask = basic_ap_mask.flatten().astype(np.int32)
						basic_pixel_ap = average_precision_score(basic_ap_mask, basic_ap_seg)
						return auroc, pixel_ap, basic_auroc, basic_pixel_ap, time_cost, basic_time_cost, ap_seg, basic_ap_seg
					return auroc, pixel_ap, None, None, time_cost, None, ap_seg, None
				# (k, i), (k+0.1, i), (k-0.1, i), (k, i+0.1), (k, i-0.1)
				best_current_auroc = 0
				best_current_ap = 0
				best_current_ap_seg = 0
				best_current_basic_ap_seg = 0
				best_current_time_cost = 0
				best_current_basic_time_cost = 0
				best_current_basic_auroc = 0
				best_current_basic_ap = 0
				best_scores = None
				best_basic_scores = None
				task_name = "mvtec"
				if dataset_name.startswith("visa"):
					task_name = "visa"
				current_auroc, current_ap, current_basic_auroc, current_basic_ap, time_cost, basic_time_cost, ap_seg, basic_ap_seg = calculation_eval(
					scores, segmentations, basic_scores, basic_segmentations)
				if((best_current_auroc ==0 and best_current_ap == 0) or (current_auroc > best_current_auroc and current_ap > best_current_ap) 
					or ((current_auroc > best_current_auroc) and abs(current_auroc - best_current_auroc) > abs(current_ap - best_current_ap)) 
					or ((current_ap > best_current_ap) and abs(current_auroc - best_current_auroc) < abs(current_ap - best_current_ap))):
					best_current_auroc = current_auroc
					best_current_time_cost = time_cost
					best_scores = scores
					best_current_ap = current_ap
					best_current_ap_seg = ap_seg
				if use_basic:
					if((best_current_basic_auroc ==0 and best_current_basic_ap == 0) 
				or (current_basic_auroc > best_current_basic_auroc and current_basic_ap > best_current_basic_ap) 
						or ((current_basic_auroc > best_current_basic_auroc) 
						and abs(current_basic_auroc - best_current_basic_auroc) > abs(current_basic_ap - best_current_basic_ap)) 
						or ((current_basic_ap > best_current_basic_ap) 
						and abs(current_basic_auroc - best_current_basic_auroc) < abs(current_basic_ap - best_current_basic_ap))):
						best_current_basic_ap = current_basic_ap
						best_current_basic_ap_seg = basic_ap_seg
						best_current_basic_time_cost = basic_time_cost
						best_current_basic_auroc = current_basic_auroc
						best_basic_scores = basic_scores
				print("current_auroc: ", best_current_auroc, "  current_ap: ", best_current_ap)
				if((pr_auroc == 0 and pr_ap == 0) or (best_current_auroc > pr_auroc and best_current_ap > pr_ap) 
			 		or (best_current_auroc > pr_auroc and abs(best_current_auroc - pr_auroc) > abs(best_current_ap - pr_ap)) 
			 				or (best_current_ap > pr_ap and abs(best_current_auroc - pr_auroc) < abs(best_current_ap - pr_ap))):
					# if(pr_auroc!=0):
					# 	result_collect.pop()
					memory_feature_list[dataloader_count] = memory_feature #######################保存最好的知识特征向量
					# 保存最好的文本提示
					if not only_vision:
						text_prompt_list[dataloader_count] = mucad_model.learnable_text_prompt.get_cur_prompts()
					# 保存最好的视觉提示
					prompt_list[dataloader_count] = mucad_model.PatchCore.prompt_model.visual.transformer.get_cur_prompt()###############保存最好的视觉提示
					# mucad_model.PatchCore.self_adaption_fusion_bank_args[task_name][0][dataloader_count] = best_k_1
					# mucad_model.PatchCore.self_adaption_fusion_bank_args[task_name][1][dataloader_count] = best_k_2
					pr_auroc = best_current_auroc
					pr_ap = best_current_ap
					img_ap = average_precision_score(anomaly_labels,best_scores)
					# Compute PRO score & PW Auroc for all images
					segmentations = best_current_ap_seg.reshape(-1,224,224)
					# (Optional) Plot example images.
					save_segmentation_images = True
					if save_segmentation_images:
						image_paths = [
							x[2] for x in dataloaders["testing"].dataset.data_to_iterate
						]
						mask_paths = [
							x[3] for x in dataloaders["testing"].dataset.data_to_iterate
						]

						def image_transform(image):
							in_std = np.array(
								dataloaders["testing"].dataset.transform_std
							).reshape(-1, 1, 1)
							in_mean = np.array(
								dataloaders["testing"].dataset.transform_mean
							).reshape(-1, 1, 1)
							image = dataloaders["testing"].dataset.transform_img(image)
							return np.clip(
								(image.numpy() * in_std + in_mean) * 255, 0, 255
							).astype(np.uint8)

						def mask_transform(mask):
							return dataloaders["testing"].dataset.transform_mask(mask).numpy()

						image_save_path = os.path.join(
							run_save_path, "segmentation_images_val", dataset_name
						)
						os.makedirs(image_save_path, exist_ok=True)
						patchcore.utils.plot_segmentation_images(
							image_save_path,
							image_paths,
							segmentations,
							best_scores,
							mask_paths,
							image_transform=image_transform,
							mask_transform=mask_transform,
							image_type = 'png' if dataset_name.startswith("mvtec") else 'JPG'
						)
					pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
						segmentations, masks_gt
					)
					full_pixel_auroc = pixel_scores["auroc"]
					# Compute PRO score & PW Auroc only images with anomalies
					sel_idxs = []
					for i in range(len(masks_gt)):
						if np.sum(masks_gt[i]) > 0:
							sel_idxs.append(i)
					pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
						[segmentations[i] for i in sel_idxs],
						[masks_gt[i] for i in sel_idxs],
					)
					anomaly_pixel_auroc = pixel_scores["auroc"]
					for i,mask in enumerate(masks_gt):
						masks_gt[i] = np.array(mask[0])
					for i,seg in enumerate(segmentations):
						segmentations[i] = np.array(seg)
					pixel_pro, pro_curve  = calculate_au_pro(np.array(masks_gt),np.array(segmentations))
					print('current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
							(dataloader_count+1,dataloader_count+1,best_current_auroc,full_pixel_auroc,img_ap,best_current_ap,pixel_pro,time_cost))
					best_auroc,best_full_pixel_auroc,best_img_ap,best_pixel_ap,best_pixel_pro,best_time_cost = best_current_auroc,full_pixel_auroc,img_ap,best_current_ap,pixel_pro,best_current_time_cost
				if use_basic:
				# calc aupro and save metric without memory limit
					if((basic_pr_auroc == 0 and basic_pr_ap == 0) or (best_current_basic_auroc > basic_pr_auroc and best_current_basic_ap > basic_pr_ap) 
						or (best_current_basic_auroc > basic_pr_auroc and abs(best_current_basic_auroc - basic_pr_auroc) > abs(best_current_basic_ap - basic_pr_ap)) 
								or (best_current_basic_ap > basic_pr_ap and abs(best_current_basic_ap - basic_pr_ap) > abs(best_current_basic_auroc - basic_pr_auroc))):
						# if(basic_pr_auroc!=0):
						# 	result_collect_nolimit.pop()
						# mucad_model.PatchCore.self_adaption_fusion_bank_args_basic[task_name][0][dataloader_count] = best_basic_k_1
						# mucad_model.PatchCore.self_adaption_fusion_bank_args_basic[task_name][1][dataloader_count] = best_basic_k_2
						basic_pr_auroc = best_current_basic_auroc
						basic_pr_ap = best_current_basic_ap 
						basic_img_ap = average_precision_score(basic_anomaly_labels,best_basic_scores)
						# Compute PRO score & PW Auroc for all images
						basic_segmentations = best_current_basic_ap_seg.reshape(-1,224,224)
						# (Optional) Plot example images.
						
						basic_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
							basic_segmentations, basic_masks_gt
						)
						basic_full_pixel_auroc = basic_pixel_scores["auroc"]
						# Compute PRO score & PW Auroc only images with anomalies
						basic_sel_idxs = []
						for i in range(len(basic_masks_gt)):
							if np.sum(basic_masks_gt[i]) > 0:
								basic_sel_idxs.append(i)
						basic_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
							[basic_segmentations[i] for i in basic_sel_idxs],
							[basic_masks_gt[i] for i in basic_sel_idxs],
						)
						basic_anomaly_pixel_auroc = basic_pixel_scores["auroc"]
						for i,mask in enumerate(basic_masks_gt):
							basic_masks_gt[i] = np.array(mask[0])
						for i,seg in enumerate(basic_segmentations):
							basic_segmentations[i] = np.array(seg)
						basic_pixel_pro, basic_pro_curve  = calculate_au_pro(np.array(basic_masks_gt),np.array(basic_segmentations))
						print('Nolimlit current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
											(dataloader_count+1,dataloader_count+1,best_current_basic_auroc,basic_full_pixel_auroc,basic_img_ap,best_current_basic_ap,basic_pixel_pro,best_current_basic_time_cost))
						best_basic_auroc,best_basic_full_pixel_auroc,best_basic_img_ap,best_basic_pixel_ap,best_basic_pixel_pro,best_basic_time_cost = best_current_basic_auroc,basic_full_pixel_auroc,basic_img_ap,best_current_basic_ap,basic_pixel_pro,best_current_basic_time_cost
			# adaption_sigmoid
			op_predict(model=mucad_model, dataloader_count=dataloader_count,
					 dataloaders=dataloaders, text_prompt_len=text_prompt_len,
					 multi_layer=multi_layer, prompt=prompt_list[dataloader_count], text_prompt=text_prompt_list[dataloader_count],
					 memory_feature=memory_feature_list[dataloader_count],
					 run_save_path=run_save_path, use_adapt_b_by_grad_down=True, task_name=task_name) 
			# inference process
			mucad_model.eval()
			predict_res = predict(model=mucad_model, dataloader_count=dataloader_count,
					 dataloaders=dataloaders, text_prompt_len=text_prompt_len,
					 multi_layer=multi_layer, prompt=prompt_list[dataloader_count], text_prompt=text_prompt_list[dataloader_count],
					 memory_feature=memory_feature_list[dataloader_count],
					 run_save_path=run_save_path, use_adapt_b_by_grad_down=True)
			result_collect.append(
						{
							"dataset_name": predict_res["dataset_name"],
							"instance_auroc": predict_res["instance_auroc"],
							"full_pixel_auroc": predict_res["full_pixel_auroc"],
							"anomaly_pixel_auroc": predict_res["anomaly_pixel_auroc"],
							"image_ap": predict_res["image_ap"],
							"pixel_ap": predict_res["pixel_ap"],
							"pixel_pro": predict_res["pixel_pro"],
							"time_cost:": predict_res["time_cost"]
						}
					)
			# 测试集
			# print('Limited current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
			# 		(dataloader_count+1,dataloader_count+1,best_auroc,best_full_pixel_auroc,best_img_ap,best_pixel_ap,best_pixel_pro,best_time_cost))
			# if use_basic:
			# 	print('Nolimlited current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
			# 		(dataloader_count+1,dataloader_count+1,best_basic_auroc,best_basic_full_pixel_auroc,best_basic_img_ap,best_basic_pixel_ap,best_basic_pixel_pro,best_basic_time_cost))

			if save_patchcore_model:
				patchcore_save_path = os.path.join(
					run_save_path, "models", dataset_name
				)
				os.makedirs(patchcore_save_path, exist_ok=True)
				for i, PatchCore in enumerate([mucad_model.PatchCore]):
					prepend = (
						"Ensemble-{}-{}_".format(i + 1, len([mucad_model.PatchCore]))
						if len([mucad_model.PatchCore]) > 1
						else ""
					)
					PatchCore.save_to_path(patchcore_save_path, prepend)


	# # Store all results and mean scores to a csv-file.
	# limited result
	print('Average result with limited')
	result_metric_names = list(result_collect[-1].keys())[1:]
	result_dataset_names = [results["dataset_name"] for results in result_collect]
	result_scores = [list(results.values())[1:] for results in result_collect]
	patchcore.utils.compute_and_store_final_results(
		run_save_path,
		result_scores,
		column_names=result_metric_names,
		row_names=result_dataset_names,
	)
	if use_basic:
		print('Average result without limited memory')
		basic_result_metric_names = list(result_collect_nolimit[-1].keys())[1:]
		basic_result_dataset_names = [results["dataset_name"] for results in result_collect_nolimit]
		basic_result_scores = [list(results.values())[1:] for results in result_collect_nolimit]
		patchcore.utils.compute_and_store_final_results(
			run_save_path_nolimit,
			basic_result_scores,
			column_names=basic_result_metric_names,
			row_names=basic_result_dataset_names,
		)
	
	# print('Average result with limited test predict')
	# result_metric_names = list(result_collect_test_predict[-1].keys())[1:]
	# result_dataset_names = [results["dataset_name"] for results in result_collect_test_predict]
	# ucad_result_scores = [list(results.values())[1:] for results in result_collect_test_predict]
	# patchcore.utils.compute_and_store_final_results(
	# 	run_save_path,
	# 	ucad_result_scores,
	# 	column_names=result_metric_names,
	# 	row_names=result_dataset_names,
	# 	predict=True
	# )


def op_predict(model, dataloader_count, dataloaders,
						text_prompt_len, multi_layer, 
						prompt, text_prompt, memory_feature,run_save_path, training=False, 
						aggregator=None, save_segmentation_images=False, use_adapt_b_by_grad_down = True, task_name='mvtec'):
	mucad_model=model
	PatchCore = mucad_model.PatchCore
	mucad_model.eval()
	mucad_model.PatchCore.adaption_sigmoid_list[task_name][0][dataloader_count].train()
	sigmoid_paramter_list = []
	for p in mucad_model.PatchCore.adaption_sigmoid_list[task_name][0][dataloader_count].parameters():
		sigmoid_paramter_list.append(p)
	for p in mucad_model.PatchCore.adaption_sigmoid_list[task_name][1][dataloader_count].parameters():
		sigmoid_paramter_list.append(p)
	optim_sigmoid = torch.optim.Adam(sigmoid_paramter_list, lr=0.5)
	dataset_name = dataloaders["training"].name
	if aggregator is None:
		aggregator = {"scores": [], "segmentations": []}
	stride = {0, 0.1, 0.5, 1, 3, -0.1, -0.5, -1, -3}
	basic_end_time = time.time()
	PatchCore.set_dataloadercount(dataloader_count)
	mucad_model.PatchCore.prompt_model.visual.transformer.set_cur_prompt(prompt, save_status=training)
	mucad_model.learnable_text_prompt.set_prompts(text_prompt, save_status=training)
	PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
	# 获取文本
	texts, prompt_idxs = get_texts(prompt_len=text_prompt_len, dataset_count=dataloader_count, only_text=False)
	# 得到文本特征
	text_feat = mucad_model(texts, prompt_idxs=prompt_idxs, use_adapter=True, use_prompt_for_wise_layer=True) # 1*768
	for ep in range(40):
		optim_sigmoid.zero_grad()
		scores, segmentations, labels_gt, masks_gt, loss = PatchCore.predict_prompt(
			dataloaders["training"], text_feat, 
			multi_layer, dataset_name=dataset_name, query_task_id=dataloader_count,
			is_predict = False,use_adapt_b_by_grad_down=True
		)
		print(f"Epoch {ep}: b={mucad_model.PatchCore.adaption_sigmoid_list[task_name][0][dataloader_count].b:.3f}")
		if np.mean(segmentations) <= 0.05:
			break
		loss.backward()
		optim_sigmoid.step()

def predict(model, dataloader_count, dataloaders,
						text_prompt_len, multi_layer, 
						prompt, text_prompt, memory_feature,run_save_path, training=False, 
						aggregator=None, save_segmentation_images=True, use_adapt_b_by_grad_down=True):
	mucad_model=model
	PatchCore = mucad_model.PatchCore
	PatchCore.prompt_model.eval()
	dataset_name = dataloaders["training"].name
	mucad_model.eval()
	if aggregator is None:
		aggregator = {"scores": [], "segmentations": []}
	basic_end_time = time.time()
	PatchCore.set_dataloadercount(dataloader_count)
	mucad_model.PatchCore.prompt_model.visual.transformer.set_cur_prompt(prompt, save_status=training)
	mucad_model.learnable_text_prompt.set_prompts(text_prompt, save_status=training)
	PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
	# 获取文本
	texts, prompt_idxs = get_texts(prompt_len=text_prompt_len, dataset_count=dataloader_count, only_text=False)
	# 得到文本特征
	text_feat = mucad_model(texts, prompt_idxs=prompt_idxs, use_adapter=True, use_prompt_for_wise_layer=True) # 1*768
	scores, segmentations, labels_gt, masks_gt, _ = PatchCore.predict_prompt(
		dataloaders["testing"], text_feat, 
		multi_layer, dataset_name=dataset_name, query_task_id=dataloader_count,
		is_predict = True, use_adapt_b_by_grad_down = use_adapt_b_by_grad_down,
	)
	# aggregator 
	aggregator["scores"].append(scores)
	aggregator["segmentations"].append(segmentations)
	end_time = time.time()	
	# 评估
	scores = np.array(aggregator["scores"])
	min_scores = scores.min(axis=-1).reshape(-1, 1)
	max_scores = scores.max(axis=-1).reshape(-1, 1)
	scores = (scores - min_scores) / (max_scores - min_scores)
	scores = np.mean(scores, axis=0)
	segmentations = np.array(aggregator["segmentations"])
	min_scores = (
		segmentations.reshape(len(segmentations), -1)
		.min(axis=-1)
		.reshape(-1, 1, 1, 1)
	)
	max_scores = (
		segmentations.reshape(len(segmentations), -1)
		.max(axis=-1)
		.reshape(-1, 1, 1, 1)
	)
	segmentations = (segmentations - min_scores) / (max_scores - min_scores)
	segmentations = np.mean(segmentations, axis=0)
	time_cost = (end_time - basic_end_time)/len(dataloaders["testing"])
	if dataset_name.startswith("mvtec"):
		anomaly_labels = [
			x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
		]
	elif dataset_name.startswith("visa"):
		anomaly_labels = [
			x[1] != "normal" for x in dataloaders["testing"].dataset.data_to_iterate
		]
	ap_seg = np.asarray(segmentations)
	ap_seg = ap_seg.flatten()
	auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
		scores, anomaly_labels
	)["auroc"]
	ap_mask = np.asarray(masks_gt)
	ap_mask = ap_mask.flatten().astype(np.int32)
	pixel_ap = average_precision_score(ap_mask,ap_seg)
	img_ap = average_precision_score(anomaly_labels,scores)
	# Compute PRO score & PW Auroc for all images
	segmentations = ap_seg.reshape(-1,224,224)
	if save_segmentation_images:
		image_paths = [
			x[2] for x in dataloaders["testing"].dataset.data_to_iterate
		]
		mask_paths = [
			x[3] for x in dataloaders["testing"].dataset.data_to_iterate
		]

		def image_transform(image):
			in_std = np.array(
				dataloaders["testing"].dataset.transform_std
			).reshape(-1, 1, 1)
			in_mean = np.array(
				dataloaders["testing"].dataset.transform_mean
			).reshape(-1, 1, 1)
			image = dataloaders["testing"].dataset.transform_img(image)
			return np.clip(
				(image.numpy() * in_std + in_mean) * 255, 0, 255
			).astype(np.uint8)

		def mask_transform(mask):
			return dataloaders["testing"].dataset.transform_mask(mask).numpy()

		image_save_path = os.path.join(
			run_save_path, "segmentation_images", dataset_name
		)
		os.makedirs(image_save_path, exist_ok=True)
		patchcore.utils.plot_segmentation_images(
			image_save_path,
			image_paths,
			segmentations,
			scores,
			mask_paths,
			image_transform=image_transform,
			mask_transform=mask_transform,
			image_type = 'png' if dataset_name.startswith("mvtec") else 'JPG'
		)
					
	pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
		segmentations, masks_gt
	)
	full_pixel_auroc = pixel_scores["auroc"]
	# Compute PRO score & PW Auroc only images with anomalies
	sel_idxs = []
	for i in range(len(masks_gt)):
		if np.sum(masks_gt[i]) > 0:
			sel_idxs.append(i)
	pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
		[segmentations[i] for i in sel_idxs],
		[masks_gt[i] for i in sel_idxs],
	)
	anomaly_pixel_auroc = pixel_scores["auroc"]
	for i,mask in enumerate(masks_gt):
		masks_gt[i] = np.array(mask[0])
	for i,seg in enumerate(segmentations):
		segmentations[i] = np.array(seg)
	pixel_pro, pro_curve  = calculate_au_pro(np.array(masks_gt),np.array(segmentations))
	print('Limited current task:{}/train task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, pixel_pro:{}, time_cost:{}'.format
					(dataloader_count+1,dataloader_count+1,auroc,full_pixel_auroc,img_ap,pixel_ap,pixel_pro,time_cost))
	return {
			"dataset_name": dataset_name,
			"instance_auroc": auroc,
			"full_pixel_auroc": full_pixel_auroc,
			"anomaly_pixel_auroc": anomaly_pixel_auroc,
			"image_ap": img_ap,
			"pixel_ap": pixel_ap,
			"pixel_pro": pixel_pro,
			"time_cost": time_cost,
			"scores": scores,
			"segmentations": ap_seg
		}


@main.command("ucad")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
	backbone_names,
	layers_to_extract_from,
	pretrain_embed_dimension,
	target_embed_dimension,
	preprocessing,
	aggregation,
	patchsize,
	patchscore,
	patchoverlap,
	anomaly_scorer_num_nn,
	patchsize_aggregate,
	faiss_on_gpu,
	faiss_num_workers,
):
	backbone_names = list(backbone_names)
	if len(backbone_names) > 1:
		layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
		for layer in layers_to_extract_from:
			idx = int(layer.split(".")[0])
			layer = ".".join(layer.split(".")[1:])
			layers_to_extract_from_coll[idx].append(layer)
	else:
		layers_to_extract_from_coll = [layers_to_extract_from]

	def get_patchcore(input_shape, sampler, device):
		loaded_patchcores = []
		for backbone_name, layers_to_extract_from in zip(
			backbone_names, layers_to_extract_from_coll
		):
			backbone_seed = None
			if ".seed-" in backbone_name:
				backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
					backbone_name.split("-")[-1]
				)
			# /home/xr/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth
			backbone = patchcore.backbones.load(backbone_name)
			backbone.name, backbone.seed = backbone_name, backbone_seed

			nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

			patchcore_instance = patchcore.patchcore.PatchCore(device)
			patchcore_instance.load(
				backbone=backbone,
				layers_to_extract_from=layers_to_extract_from,
				device=device,
				input_shape=input_shape,
				pretrain_embed_dimension=pretrain_embed_dimension,
				target_embed_dimension=target_embed_dimension,
				patchsize=patchsize,
				featuresampler=sampler,
				anomaly_scorer_num_nn=anomaly_scorer_num_nn,
				nn_method=nn_method,
			)
			loaded_patchcores.append(patchcore_instance)
		return loaded_patchcores

	return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
	def get_sampler(device):
		if name == "identity":
			return patchcore.sampler.IdentitySampler()
		elif name == "greedy_coreset":
			return patchcore.sampler.GreedyCoresetSampler(percentage, device)
		elif name == "approx_greedy_coreset":
			return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

	return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=224, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--csv_path", default=None, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
	name,
	data_path,
	subdatasets,
	train_val_split,
	batch_size,
	resize,
	imagesize,
	num_workers,
	csv_path,
	augment,
	test_val_split=1.0,
):
	dataset_info = _DATASETS[name]
	dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

	def get_dataloaders(seed):
		dataloaders = []
		for subdataset in subdatasets:
			train_dataset = dataset_library.__dict__[dataset_info[1]](
				data_path,
				classname=subdataset,
				resize=resize,
				train_val_split=train_val_split,
				imagesize=imagesize,
				split=dataset_library.DatasetSplit.TRAIN,
				csv_path=csv_path,
				seed=seed,
				augment=augment,
			)

			test_dataset = dataset_library.__dict__[dataset_info[1]](
				data_path,
				classname=subdataset,
				resize=resize,
				imagesize=imagesize,
				split=dataset_library.DatasetSplit.TEST,
				csv_path=csv_path,
				seed=seed,
			)

			train_dataloader = torch.utils.data.DataLoader(
				train_dataset,
				batch_size=batch_size,
				shuffle=True,
				num_workers=num_workers,
				pin_memory=True,
			)

			test_dataloader = torch.utils.data.DataLoader(
				test_dataset,
				batch_size=batch_size,
				shuffle=False,
				num_workers=num_workers,
				pin_memory=True,
			)

			train_dataloader.name = name
			if subdataset is not None:
				train_dataloader.name += "_" + subdataset

			if train_val_split < 1:
				val_dataset = dataset_library.__dict__[dataset_info[1]](
					data_path,
					classname=subdataset,
					resize=resize,
					train_val_split=train_val_split,
					imagesize=imagesize,
					split=dataset_library.DatasetSplit.VAL,
					seed=seed,
				)

				val_dataloader = torch.utils.data.DataLoader(
					val_dataset,
					batch_size=batch_size,
					shuffle=False,
					num_workers=num_workers,
					pin_memory=True,
				)
			else:
				val_dataloader = None

			# if test_val_split < 1:
			# val_dataset = dataset_library.__dict__[dataset_info[1]](
			# 	data_path,
			# 	classname=subdataset,
			# 	resize=resize,
			# 	test_val_split=test_val_split,
			# 	imagesize=imagesize,
			# 	split=dataset_library.DatasetSplit.VAL,
			# 	csv_path=csv_path,
			# 	seed=seed,
			# )

			# val_dataloader = torch.utils.data.DataLoader(
			# 	val_dataset,
			# 	batch_size=batch_size,
			# 	shuffle=False,
			# 	num_workers=num_workers,
			# 	pin_memory=True,
			# )
			
			# else:
			# 	val_dataloader = None
			
			dataloader_dict = {
				"training": train_dataloader,
				"validation": val_dataloader,
				"testing": test_dataloader,
			}

			dataloaders.append(dataloader_dict)
		return dataloaders

	return ("get_dataloaders", get_dataloaders)

class GroundTruthComponent:
	"""
	Stores sorted anomaly scores of a single ground truth component.
	Used to efficiently compute the region overlap for many increasing thresholds.
	"""

	def __init__(self, anomaly_scores):
		"""
		Initialize the module.

		Args:
			anomaly_scores: List of all anomaly scores within the ground truth
							component as numpy array.
		"""
		# Keep a sorted list of all anomaly scores within the component.
		self.anomaly_scores = anomaly_scores.copy()
		self.anomaly_scores.sort()

		# Pointer to the anomaly score where the current threshold divides the component into OK / NOK pixels.
		self.index = 0

		# The last evaluated threshold.
		self.last_threshold = None

	def compute_overlap(self, threshold):
		"""
		Compute the region overlap for a specific threshold.
		Thresholds must be passed in increasing order.

		Args:
			threshold: Threshold to compute the region overlap.

		Returns:
			Region overlap for the specified threshold.
		"""
		if self.last_threshold is not None:
			assert self.last_threshold <= threshold

		# Increase the index until it points to an anomaly score that is just above the specified threshold.
		while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
			self.index += 1

		# Compute the fraction of component pixels that are correctly segmented as anomalous.
		return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
	"""
	This function calculates the definit integral of a curve given by x- and corresponding y-values.
	In contrast to, e.g., 'numpy.trapz()', this function allows to define an upper bound to the integration range by
	setting a value x_max.

	Points that do not have a finite x or y value will be ignored with a warning.

	Args:
		x:     Samples from the domain of the function to integrate need to be sorted in ascending order. May contain
				 the same value multiple times. In that case, the order of the corresponding y values will affect the
				 integration with the trapezoidal rule.
		y:     Values of the function corresponding to x values.
		x_max: Upper limit of the integration. The y value at max_x will be determined by interpolating between its
				 neighbors. Must not lie outside of the range of x.

	Returns:
		Area under the curve.
	"""

	x = np.array(x)
	y = np.array(y)
	finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
	if not finite_mask.all():
		print(
			"""WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
	x = x[finite_mask]
	y = y[finite_mask]

	# Introduce a correction term if max_x is not an element of x.
	correction = 0.
	if x_max is not None:
		if x_max not in x:
			# Get the insertion index that would keep x sorted after np.insert(x, ins, x_max).
			ins = bisect(x, x_max)
			# x_max must be between the minimum and the maximum, so the insertion_point cannot be zero or len(x).
			assert 0 < ins < len(x)

			# Calculate the correction term which is the integral between the last x[ins-1] and x_max. Since we do not
			# know the exact value of y at x_max, we interpolate between y[ins] and y[ins-1].
			y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
			correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

		# Cut off at x_max.
		mask = x <= x_max
		x = x[mask]
		y = y[mask]

	# Return area under the curve using the trapezoidal rule.
	return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
	"""
	Extract anomaly scores for each ground truth connected component as well as anomaly scores for each potential false
	positive pixel from anomaly maps.

	Args:
		anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

		ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
							 for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
							 an anomaly.

	Returns:
		ground_truth_components: A list of all ground truth connected components that appear in the dataset. For each
								 component, a sorted list of its anomaly scores is stored.

		anomaly_scores_ok_pixels: A sorted list of anomaly scores of all anomaly-free pixels of the dataset. This list
									can be used to quickly select thresholds that fix a certain false positive rate.
	"""
	# Make sure an anomaly map is present for each ground truth map.
	assert len(anomaly_maps) == len(ground_truth_maps)

	# Initialize ground truth components and scores of potential fp pixels.
	ground_truth_components = []
	anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

	# Structuring element for computing connected components.
	structure = np.ones((3, 3), dtype=int)

	# Collect anomaly scores within each ground truth region and for all potential fp pixels.
	ok_index = 0
	for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):

		# Compute the connected components in the ground truth map.
		labeled, n_components = label(gt_map, structure)

		# Store all potential fp scores.
		num_ok_pixels = len(prediction[labeled == 0])
		anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
		ok_index += num_ok_pixels

		# Fetch anomaly scores within each GT component.
		for k in range(n_components):
			component_scores = prediction[labeled == (k + 1)]
			ground_truth_components.append(GroundTruthComponent(component_scores))

	# Sort all potential false positive scores.
	anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
	anomaly_scores_ok_pixels.sort()

	return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
	"""
	Compute the PRO curve at equidistant interpolation points for a set of anomaly maps with corresponding ground
	truth maps. The number of interpolation points can be set manually.

	Args:
		anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

		ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
							 for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
							 an anomaly.

		num_thresholds:    Number of thresholds to compute the PRO curve.
	Returns:
		fprs: List of false positive rates.
		pros: List of correspoding PRO values.
	"""
	# Fetch sorted anomaly scores.
	ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

	# Select equidistant thresholds.
	threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

	fprs = [1.0]
	pros = [1.0]
	for pos in threshold_positions:
		threshold = anomaly_scores_ok_pixels[pos]

		# Compute the false positive rate for this threshold.
		fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

		# Compute the PRO value for this threshold.
		pro = 0.0
		for component in ground_truth_components:
			pro += component.compute_overlap(threshold)
		pro /= len(ground_truth_components)

		fprs.append(fpr)
		pros.append(pro)

	# Return (FPR/PRO) pairs in increasing FPR order.
	fprs = fprs[::-1]
	pros = pros[::-1]

	return fprs, pros


def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
	"""
	Compute the area under the PRO curve for a set of ground truth images and corresponding anomaly images.
	Args:
		gts:         List of tensors that contain the ground truth images for a single dataset object.
		predictions: List of tensors containing anomaly images for each ground truth image.
		integration_limit:    Integration limit to use when computing the area under the PRO curve.
		num_thresholds:       Number of thresholds to use to sample the area under the PRO curve.

	Returns:
		au_pro:    Area under the PRO curve computed up to the given integration limit.
		pro_curve: PRO curve values for localization (fpr,pro).
	"""
	# Compute the PRO curve.
	pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)

	# Compute the area under the PRO curve.
	au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
	au_pro /= integration_limit

	# Return the evaluation metrics.
	return au_pro, pro_curve
'''
# def predict(model, dataloader_count, dataloaders, seed, device, device_context, methods,
# 						text_prompt_len, multi_layer,  list_of_dataloaders,
# 						prompt, text_prompt, memory_feature, training=False):
# 	mucad_model = model
# 	start_time = time.time()
# 	patchcore.utils.fix_seeds(seed, device)
# 	dataset_name = dataloaders["training"].name
# 	with device_context:
# 		predict_aggregator = {"scores": [], "segmentations": []}
# 		torch.cuda.empty_cache()
# 		for i, PatchCore in enumerate([mucad_model.PatchCore]):
# 			torch.cuda.empty_cache()
# 			if PatchCore.backbone.seed is not None:
# 					patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
# 			torch.cuda.empty_cache()
# 			PatchCore.set_dataloadercount(dataloader_count)
# 			mucad_model.PatchCore.prompt_model.visual.transformer.set_cur_prompt(prompt, save_status=training)
# 			mucad_model.learnable_text_prompt.set_prompts(text_prompt, save_status=training)
# 			mucad_model.eval()
# 			PatchCore.prompt_model.eval()
# 			PatchCore.anomaly_scorer.fit(detection_features=[memory_feature])
# 			texts, prompt_idxs = get_texts(prompt_len=text_prompt_len, dataset_count=dataloader_count, only_text=False)
# 			# 得到文本特征
# 			text_feat = mucad_model(texts, prompt_idxs=prompt_idxs, use_adapter=True, use_prompt_for_wise_layer=True) # 1*768
# 			predict_scores, predict_segmentations, predict_labels_gt, predict_masks_gt = PatchCore.predict_prompt(
# 					dataloaders["testing"], text_feat, multi_layer, dataset_name, dataloader_count
# 			)
# 			predict_aggregator["scores"].append(predict_scores)
# 			predict_aggregator["segmentations"].append(predict_segmentations)		
# 		predict_scores = np.array(predict_aggregator["scores"])
# 		predict_min_scores = predict_scores.min(axis=-1).reshape(-1, 1)
# 		predict_max_scores = predict_scores.max(axis=-1).reshape(-1, 1)
# 		predict_scores = (predict_scores - predict_min_scores) / (predict_max_scores - predict_min_scores)
# 		predict_scores = np.mean(predict_scores, axis=0)
# 		predict_segmentations = np.array(predict_aggregator["segmentations"])
# 		predict_min_scores = (
# 				predict_segmentations.reshape(len(predict_segmentations), -1)
# 				.min(axis=-1)
# 				.reshape(-1, 1, 1, 1)
# 		)
# 		predict_max_scores = (
# 				predict_segmentations.reshape(len(predict_segmentations), -1)
# 				.max(axis=-1)
# 				.reshape(-1, 1, 1, 1)
# 		)
# 		predict_segmentations = (predict_segmentations - predict_min_scores) / (predict_max_scores - predict_min_scores)
# 		predict_segmentations = np.mean(predict_segmentations, axis=0)
# 		end_time = time.time()
# 		time_cost = (end_time - start_time)/len(dataloaders["testing"])
# 		if dataset_name.startswith("mvtec"):
# 			predict_anomaly_labels = [
# 				x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
# 			]
# 		elif dataset_name.startswith("visa"):
# 			predict_anomaly_labels = [
# 				x[1] != "normal" for x in dataloaders["testing"].dataset.data_to_iterate
# 			]
# 		predict_ap_seg = np.asarray(predict_segmentations)
# 		predict_ap_seg = predict_ap_seg.flatten()
# 		predict_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
# 				predict_scores, predict_anomaly_labels
# 		)["auroc"]
# 		predict_ap_mask = np.asarray(predict_masks_gt)
# 		predict_ap_mask = predict_ap_mask.flatten().astype(np.int32)
# 		predict_pixel_ap = average_precision_score(predict_ap_mask,predict_ap_seg)
# 		predict_img_ap = average_precision_score(predict_anomaly_labels,predict_scores)
# 		predict_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
# 			predict_segmentations, predict_masks_gt
# 		)
# 		predict_full_pixel_auroc = predict_pixel_scores["auroc"]
# 		# Compute PRO score & PW Auroc only images with anomalies
# 		predict_sel_idxs = []
# 		for i in range(len(predict_masks_gt)):
# 			if np.sum(predict_masks_gt[i]) > 0:
# 				predict_sel_idxs.append(i)
# 		predict_pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
# 			[predict_segmentations[i] for i in predict_sel_idxs],
# 			[predict_masks_gt[i] for i in predict_sel_idxs],
# 		)
# 		predict_anomaly_pixel_auroc = predict_pixel_scores["auroc"]
# 		for i,mask in enumerate(predict_masks_gt):
# 			predict_masks_gt[i] = np.array(mask[0])
# 		for i,seg in enumerate(predict_segmentations):
# 			predict_segmentations[i] = np.array(seg)
# 		predict_pixel_pro, predict_pro_curve  = calculate_au_pro(np.array(predict_masks_gt),np.array(predict_segmentations))	
	
# 	return {
# 				"dataset_name": dataset_name,
# 				"instance_auroc": predict_auroc,
# 				"full_pixel_auroc": predict_full_pixel_auroc,
# 				"anomaly_pixel_auroc": predict_anomaly_pixel_auroc,
# 				"image_ap": predict_img_ap,
# 				"pixel_ap": predict_pixel_ap,
# 				"pixel_pro": predict_pixel_pro,
# 				"time_cost": time_cost,
# 				"scores": predict_scores,
# 				"segmentations": predict_ap_seg.reshape(-1, 224, 224)
# 			}
	# return {
	# 			"dataset_name": dataset_name,
	# 			"instance_auroc": 1,
	# 			"full_pixel_auroc": 1,
	# 			"anomaly_pixel_auroc": 1,
	# 			"image_ap": 1,
	# 			"pixel_ap": 1,
	# 			"pixel_pro": 1,
	# 			"time_cost": 1
	# 		}
'''

if __name__ == "__main__":
	# parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
	# config = parser.parse_known_args()[-1][0]
	# subparser = parser.add_subparsers(dest='subparser_name')

	# from patchcore.configs.mvtecad_dualprompt import get_args_parser
	# config_parser = subparser.add_parser('mvtecad_dualprompt', help='MVTec AD')
	# get_args_parser(config_parser)
	# args = parser.parse_args()
	# print(args)

	logging.basicConfig(level=logging.INFO)
	LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
	main()
