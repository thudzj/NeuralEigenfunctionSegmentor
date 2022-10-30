pip install mmcv==1.3.8 mmsegmentation==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install kmeans-pytorch
pip install timm==0.4.12

first download the dataset to some dir
```
export DATASET=~/data
python -m scripts.prepare_ade20k $DATASET
```

then train our model
```
CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/normalized_adjacency-l2 --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel normalized_adjacency --input_l2_normalize --port 12345

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/normalized_adjacency --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel normalized_adjacency --no-input_l2_normalize --port 12346

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/normalized_adjacency-l2-relu --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel normalized_adjacency --input_l2_normalize --psi_norm_type idt --psi_act_type relu --port 12347

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/normalized_adjacency-l2-bnrelu --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel normalized_adjacency --input_l2_normalize --psi_norm_type bn --psi_act_type relu --port 12348

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/linear-l2 --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel linear --input_l2_normalize --port 12349

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/linear --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel linear --no-input_l2_normalize --port 12350

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/linear-l2-bnrelu --dataset ade20k --backbone vit_small_patch16_384 --no-resume --batch-size 16 --weight-decay 0 --optimizer adamw --scheduler cosine --learning-rate 0.001 --epochs 10 --kernel linear --input_l2_normalize  --psi_norm_type bn --psi_act_type relu --port 12351

```

and test with
```
CUDA_VISIBLE_DEVICES=1 python inference.py --model-path logs/normalized_adjacency-l2/checkpoint.pth -i $DATASET/ade20k/ADEChallengeData2016/images/validation -o segmaps/ 
```