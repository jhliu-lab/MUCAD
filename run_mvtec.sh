#!/bin/bash
source /root/anaconda3/bin/activate
conda activate pytorch1.12.1_ljh_mucad_1
datapath=/home/ljh/datasets/mvtec2d
# datasets=('cable' 'capsule' 'carpet'  'hazelnut' 
# 'leather' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'wood')
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
CUDA_VISIBLE_DEVICES=0  python3 run_mucad.py --gpu 0 --seed 0 --memory_size 1960 --log_group mvtec_1960_50epoch_test --save_segmentation_images --log_project mvtec  second_paper  ucad -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 1 sampler -p 0.1 approx_greedy_coreset dataset --resize 224 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath 
# memery_bank_vision_prompt_text_prompt_24-vision_layers_3-noise_level_0.1_0.5-50_epoch
# memery_bank-alpha_0.5-vision_prompt_text_prompt_24-vision_layers_3-noise_level_0.1_0.5-50_epoch_new_noise                      mvtec Test_res
# memery_bank-alpha_0.5-vision_prompt_text_prompt_24-vision_layers_3-noise_ratio_1.0-50_epoch_new_noise_53.8
# memery_bank-alpha_0.5-vision_prompt_text_prompt_24-vision_layers_3-noise_ratio_1.0-50_epoch_new_noise_new_sigmoid_b_10
# memery_bank_alpha_0.5-vison_prompt_text_prompt_24_vison_layers-3-noise_level_1.0_fusion_model_50_epoch    
# new_res mvtec_resultssupplementary_experiments