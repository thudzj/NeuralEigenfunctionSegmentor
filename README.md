```
pip install mmcv==1.3.8 mmsegmentation==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==0.4.12
pip install fast-pytorch-kmeans
```

first download the dataset to some dir
```
export DATASET=~/data
python -m scripts.prepare_ade20k $DATASET
```

then train our model
```
----
CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/8 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --no-input_l2_normalize --port 12347
    loss: 1.3360 (1.3743)

---- 
CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/12 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --input_l2_normalize --port 11111
    loss: 1.2991 (1.3680)

----
CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/15 --dataset ade20k --no-resume --batch-size 16 --epochs 30 --input_l2_normalize --port 11114
    loss: 1.3804 (1.3100)

----
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/21 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11110 --upsample_factor 2 --cache_size 30

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/22 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11111 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 0.1

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/28 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11117 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 0.2

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/23 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11112 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 0.5

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/24 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11113 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 1

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/25 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11114 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 2

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/26 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11115 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 4

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/27 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11116 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 8


--pixelwise_adj_div_factor 255

```

and test with
```
CUDA_VISIBLE_DEVICES=1 python inference.py --model-path logs/normalized_adjacency-l2/checkpoint.pth -i $DATASET/ade20k/ADEChallengeData2016/images/validation -o segmaps/ 
```