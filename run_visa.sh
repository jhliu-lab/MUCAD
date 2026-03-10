#!/bin/bash
source /root/anaconda3/bin/activate
conda activate pytorch1.12.1_ljh_mucad_1
datapath=/home/ljh/datasets/visa
datasets=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1'
'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
CUDA_VISIBLE_DEVICES=1 python3 run_mucad.py --gpu 0 --seed 0 --memory_size 9800 --log_group visa_9800_50epoch_test  --save_segmentation_images --log_project visa  second_paper ucad -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 1 sampler -p 0.1 approx_greedy_coreset dataset --resize 224 --imagesize 224 --csv_path /home/ljh/datasets/visa/split_csv/1cls.csv  "${dataset_flags[@]}" visa $datapath 
