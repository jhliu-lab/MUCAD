#!/bin/bash
source /root/anaconda3/bin/activate
conda activate pytorch1.12.1_ljh_mucad_1
datapath=/home/ljh/datasets/mvtec2d
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
CUDA_VISIBLE_DEVICES=0  python3 run_mucad.py --gpu 0 --seed 0 --memory_size 1960 --log_group mvtec_1960_50epoch_test --save_segmentation_images --log_project mvtec  second_paper  ucad -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 1 sampler -p 0.1 approx_greedy_coreset dataset --resize 224 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath 
