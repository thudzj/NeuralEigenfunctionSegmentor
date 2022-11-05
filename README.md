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
CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/8 --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.01 --epochs 20 --kernel normalized_adjacency_thresholded_cache_for_D --no-input_l2_normalize --port 12347 --psi_norm_type bn --psi_act_type gelu --alpha 2
    loss: 1.3360 (1.3743)

---- 
CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/12 --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.01 --epochs 20 --kernel normalized_adjacency_thresholded_cache_for_D --input_l2_normalize --port 11111 --psi_norm_type bn --psi_act_type gelu --alpha 2
    loss: 1.2991 (1.3680)

----

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/15 --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.01 --epochs 30 --kernel normalized_adjacency_thresholded_cache_for_D --input_l2_normalize --port 11114 --psi_norm_type bn --psi_act_type gelu --alpha 2

----

```

and test with
```
CUDA_VISIBLE_DEVICES=1 python inference.py --model-path logs/normalized_adjacency-l2/checkpoint.pth -i $DATASET/ade20k/ADEChallengeData2016/images/validation -o segmaps/ 
```