```
pip install mmcv==1.3.8 mmsegmentation==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==0.4.12
pip install SimpleCRF
```

first download the dataset to some dir
```
export DATASET=~/data
python -m scripts.prepare_ade20k $DATASET
```

then train our model
```

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir vqseg_logs/2 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --embedding_dim 32 --n_embeddings 128 --encoder_n_layers 3 --decoder_n_layers 3 --beta 1. --recon_target highlevel_feature 
    Stats ['final']: pixel_accuracy: 36.7400 (36.7400)  mean_accuracy: 25.7200 (25.7200)  mean_iou: 13.3400 (13.3400)
```

eval
```
CUDA_VISIBLE_DEVICES=2 python train.py --log-dir vqseg_logs/2 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --embedding_dim 32 --n_embeddings 128 --encoder_n_layers 3 --decoder_n_layers 3 --beta 1. --recon_target highlevel_feature --eval-only
```