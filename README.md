# Pytorch implementation of Paper "Exploring Multimodal Prompts For Unsupervised Continuous Anomaly Detection"
# MUCAD in ACM MM-2025

## Introduction

MUCAD is a novel Unsupervised Continual AD framework.

## environment

### basic
python>=3.8, torch>=1.12, CUDA>=11.3, timm==0.6.7

### install SAM:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
or clone the repository locally and install with
```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

## prepare for training
rename the dataset dir to 'mvtec2d' and create sam senmantic dir
(processed mvtec2d-sam-b.zip is provided in repository)
```
cp -r $mvtec_origin_data_path('./mvtec2d') $mvtec_data_path('./mvtec2d-sam-b')
cd UCAD/segment_anything
python3 dataset_sam.py --sam_type 'vit_b' --sam_checkpoint $your_sam_path --data_path $mvtec_data_path
```

## training and evaluation
environment prepare:
```
datapath=/hhd3/m3lab/data/mvtec2d datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
```

<!-- datapath=/hhd3/m3lab/data/visa datasets=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum') -->
training:
```
CUDA_VISIBLE_DEVICES=0  python3 run_mucad.py --gpu 0 --seed 0 --memory_size 1960 --log_group mvtec_196_50epoch_test --save_segmentation_images --log_project mvtec  second_paper  ucad -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 1 sampler -p 0.1 approx_greedy_coreset dataset --resize 224 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath 
```

### Parameter
Main contents are contained in three files: ./patchcore/patchcore.py, ./patchcore/vision_transformer.py, and ./run_ucad.py.
Whether to save the image, the image size, and the memory size can all be modified in the above training command.
Parameters about prompt are in ./patchcore/patchcore.py line 99.

The inference involving a query process, it's slow, and I've commented it out in the code (./run_ucad.py lines 408-509). Training will directly provide the final results, and the inference process merely repeats this step. The final output will consist of two parts, with the lower metrics representing the final results, and the difference between them and the higher metrics results is denoted as FM.

## Acknowledgments

Our benchmark is built on [UCAD](https://github.com/jiaq-liu/UCAD) and [CLIP](https://github.com/openai/CLIP), thanks their extraordinary works!
