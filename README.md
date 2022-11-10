```
pip install mmcv==1.3.8 mmsegmentation==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==0.4.12
pip install fast-pytorch-kmeans
pip install SimpleCRF
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
    --kmeans_n_cls 150 Stats ['final']: pixel_accuracy: 59.4400 (59.4400)  mean_accuracy: 12.9600 (12.9600)  mean_iou: 8.1500 (8.1500)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/43 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11113 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 8
    loss: 1.2956 (1.3739)
    --kmeans_n_cls 150 Stats ['final']: pixel_accuracy: 61.4500 (61.4500)  mean_accuracy: 15.3100 (15.3100)  mean_iou: 10.2700 (10.2700)
      (mapping by miou) Stats ['final']: pixel_accuracy: 56.6600 (56.6600)  mean_accuracy: 19.9300 (19.9300)  mean_iou: 10.8300 (10.8300)

--- baseline ---
CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/baseline --dataset ade20k --no-resume --batch-size 10 --epochs 0 --port 11137 --psi_num_layers 0
    --kmeans_n_cls 150 Stats ['final']: pixel_accuracy: 56.6100 (56.6100)  mean_accuracy: 18.1300 (18.1300)  mean_iou: 10.3700 (10.3700)

-----
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/31 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11110 --upsample_factor 2 --cache_size 30 --psi_k 256

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/32 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11111 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/33 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11112 --upsample_factor 2 --cache_size 30 --psi_k 512

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/34 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11133 --upsample_factor 2 --cache_size 30 --psi_k 512 --pixelwise_adj_weight 8

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/35 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11114 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8 --psi_res --psi_num_layers 8 --psi_act_type relu

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/36 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11115 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8 --optimizer lars -lr 0.1

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/37 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11116 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8 --psi_res --psi_num_layers 8 --psi_act_type relu --optimizer lars -lr 0.1

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/38 --dataset ade20k --no-resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11117 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8 --psi_res --psi_num_layers 8 --psi_act_type relu --optimizer our_lars -lr 0.1


***
CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/34 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11114 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8
    loss: 1.2898 (1.3150)
    --kmeans_n_cls 256 Stats ['final']: pixel_accuracy: 61.7200 (61.7200)  mean_accuracy: 13.4200 (13.4200)  mean_iou: 9.5300 (9.5300)
***

----
source ~/anaconda3/bin/activate; conda activate env37; CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/40 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11110 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 1

source ~/anaconda3/bin/activate; conda activate env37; CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/41 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11111 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 2

source ~/anaconda3/bin/activate; conda activate env37; CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/42 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11112 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 4

source ~/anaconda3/bin/activate; conda activate env37; CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/43 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11113 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 8



source ~/anaconda3/bin/activate; conda activate env37; CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/45 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11115 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 32

source ~/anaconda3/bin/activate; conda activate env37; CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/46 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11116 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 64

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/47 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11117 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 0.5

```

and test with
```
CUDA_VISIBLE_DEVICES=1 python inference.py --model-path logs/normalized_adjacency-l2/checkpoint.pth -i $DATASET/ade20k/ADEChallengeData2016/images/validation -o segmaps/ 
```