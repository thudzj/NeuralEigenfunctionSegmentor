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
    Stats ['final']: pixel_accuracy: 55.9300 (55.9300)  mean_accuracy: 12.7900 (12.7900)  mean_iou: 7.5100 (7.5100)

---- 
CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/12 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --input_l2_normalize --port 11111
    loss: 1.2991 (1.3680)
    Stats ['final']: pixel_accuracy: 54.4100 (54.4100)  mean_accuracy: 12.0100 (12.0100)  mean_iou: 7.3200 (7.3200)

----
CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/15 --dataset ade20k --no-resume --batch-size 16 --epochs 30 --input_l2_normalize --port 11114
    loss: 1.3804 (1.3100)
    Stats ['final']: pixel_accuracy: 55.6900 (55.6900)  mean_accuracy: 12.9400 (12.9400)  mean_iou: 8.1900 (8.1900)

----
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/21 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11110 --upsample_factor 2 --cache_size 30
    loss: 1.4982 (1.4263)
    Stats ['final']: pixel_accuracy: 59.2500 (59.2500)  mean_accuracy: 13.4000 (13.4000)  mean_iou: 8.5300 (8.5300)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/27 --dataset ade20k --np-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11116 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 8
    loss: 1.9259 (1.9139)
    Stats ['final']: pixel_accuracy: 60.1700 (60.1700)  mean_accuracy: 14.6000 (14.6000)  mean_iou: 9.2000 (9.2000)

----
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/30 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11110 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 512

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/31 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11111 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/32 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11112 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256 --pixelwise_adj_weight 2

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/33 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11113 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256 --pixelwise_adj_weight 4

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/34 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11114 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256 --pixelwise_adj_weight 8

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/35 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11115 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256 --pixelwise_adj_weight 2 --pixelwise_adj_div_factor 255

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/36 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11116 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256 --pixelwise_adj_weight 4 --pixelwise_adj_div_factor 255

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/37 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11117 --upsample_factor 2 --cache_size 30 --kmeans_n_cls 256 --psi_k 256 --pixelwise_adj_weight 8 --pixelwise_adj_div_factor 255
```

and test with
```
CUDA_VISIBLE_DEVICES=1 python inference.py --model-path logs/normalized_adjacency-l2/checkpoint.pth -i $DATASET/ade20k/ADEChallengeData2016/images/validation -o segmaps/ 
```