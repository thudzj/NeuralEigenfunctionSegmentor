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
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/40 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    our: pixel_accuracy: 66.9400 (66.9400)  mean_accuracy: 63.3700 (63.3700)  mean_iou: 41.0300 (41.0300)
    our_kmeans: pixel_accuracy: 69.6400 (69.6400)  mean_accuracy: 59.3400 (59.3400)  mean_iou: 42.2500 (42.2500)   
    kmeans: pixel_accuracy: 56.2800 (56.2800)  mean_accuracy: 50.3100 (50.3100)  mean_iou: 31.9200 (31.9200)

    reco:
        our: mIoU (bi) 0.367 (0.371) | Pixel acc (bi) 0.688 (0.689)
        our_kmeans: mIoU (bi) 0.395 (0.395) | Pixel acc (bi) 0.744 (0.744)
        kmeans: mIoU (bi) 0.289 (0.289) | Pixel acc (bi) 0.618 (0.619)
    
    CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/40-2 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco
        our: pixel_accuracy: 67.8800 (67.8800)  mean_accuracy: 63.5900 (63.5900)  mean_iou: 41.7700 (41.7700)
        our_kmeans: pixel_accuracy: 68.1500 (68.1500)  mean_accuracy: 59.0000 (59.0000)  mean_iou: 40.2600 (40.2600)
        kmeans: pixel_accuracy: 56.2800 (56.2800)  mean_accuracy: 50.3100 (50.3100)  mean_iou: 31.9200 (31.9200)
        
        reco:
            mIoU (bi) 0.400 (0.404) | Pixel acc (bi) 0.717 (0.719)
            our_kmeans: mIoU (bi) 0.398 (0.398) | Pixel acc (bi) 0.748 (0.749)
            kmeans: mIoU (bi) 0.289 (0.289) | Pixel acc (bi) 0.618 (0.619)


CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/41 --dataset pascal_context --no-resume --backbone vit_base_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    pixel_accuracy: pixel_accuracy: 63.3100 (63.3100)  mean_accuracy: 63.3300 (63.3300)  mean_iou: 38.8900 (38.8900)  
    our_kmeans: pixel_accuracy: 66.5800 (66.5800)  mean_accuracy: 59.2700 (59.2700)  mean_iou: 39.0200 (39.0200)
    kmeans: pixel_accuracy: 51.4000 (51.4000)  mean_accuracy: 49.4700 (49.4700)  mean_iou: 28.6400 (28.6400)

    reco:
        mIoU (bi) 0.369 (0.375) | Pixel acc (bi) 0.693 (0.697)
        our_kmeans:  mIoU (bi) 0.375 (0.376) | Pixel acc (bi) 0.731 (0.732)
        kmeans: mIoU (bi) 0.309 (0.309) | Pixel acc (bi) 0.587 (0.587)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/42 --dataset pascal_context --no-resume --backbone vit_large_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    pixel_accuracy: pixel_accuracy: 59.0200 (59.0200)  mean_accuracy: 60.6800 (60.6800)  mean_iou: 36.0800 (36.0800)
    our_kmeans: pixel_accuracy: pixel_accuracy: 67.4000 (67.4000)  mean_accuracy: 53.3700 (53.3700)  mean_iou: 36.6800 (36.6800)
    kmeans: pixel_accuracy: 40.6300 (40.6300)  mean_accuracy: 40.8600 (40.8600)  mean_iou: 21.0500 (21.0500)
    
    reco:
        mIoU (bi) 0.326 (0.332) | Pixel acc (bi) 0.627 (0.632)
        our_kmeans: mIoU (bi) 0.349 (0.350) | Pixel acc (bi) 0.719 (0.719)
        kmeans: mIoU (bi) 0.193 (0.193) | Pixel acc (bi) 0.453 (0.453)


CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/43 --dataset pascal_context --no-resume --backbone vit_base_patch8_224 --batch-size 6 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    pixel_accuracy: 73.8400 (73.8400)  mean_accuracy: 56.6200 (56.6200)  mean_iou: 42.6400 (42.6400)

--------------------

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/44 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04 
    pixel_accuracy: 63.2800 (63.2800)  mean_accuracy: 39.0100 (39.0100)  mean_iou: 21.5700 (21.5700)
    our_kmeans: pixel_accuracy: 62.4900 (62.4900)  mean_accuracy: 47.1600 (47.1600)  mean_iou: 23.5800 (23.5800)
    kmeans: pixel_accuracy: 50.7000 (50.7000)  mean_accuracy: 41.3700 (41.3700)  mean_iou: 19.2200 (19.2200)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/45 --dataset ade20k --no-resume --backbone vit_base_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04
    pixel_accuracy: 61.0000 (61.0000)  mean_accuracy: 45.9800 (45.9800)  mean_iou: 25.0000 (25.0000)
    our_kmeans: pixel_accuracy: 62.2000 (62.2000)  mean_accuracy: 45.9100 (45.9100)  mean_iou: 23.3600 (23.3600)
    kmeans: pixel_accuracy: 48.6000 (48.6000)  mean_accuracy: 45.9900 (45.9900)  mean_iou: 19.6000 (19.6000)  

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/46 --dataset ade20k --no-resume --backbone vit_large_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04
    pixel_accuracy: 59.0300 (59.0300)  mean_accuracy: 44.7000 (44.7000)  mean_iou: 25.2500 (25.2500) 
    our_kmeans: pixel_accuracy: 61.1400 (61.1400)  mean_accuracy: 42.9000 (42.9000)  mean_iou: 22.5800 (22.5800)
    kmeans: pixel_accuracy: 38.3500 (38.3500)  mean_accuracy: 42.8100 (42.8100)  mean_iou: 17.8400 (17.8400)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/47 --dataset ade20k --no-resume --backbone vit_base_patch8_224 --batch-size 6 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04
    pixel_accuracy: 66.4500 (66.4500)  mean_accuracy: 24.9300 (24.9300)  mean_iou: 17.4800 (17.4800)
    our_kmeans: pixel_accuracy: 62.0500 (62.0500)  mean_accuracy: 45.0000 (45.0000)  mean_iou: 22.5200 (22.5200)

--------------------
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/50 --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    disable-crf:
        pixel_accuracy: 85.4800 (85.4800)  mean_accuracy: 66.7700 (66.7700)  mean_iou: 45.8800 (45.8800)
        our_kmeans: pixel_accuracy: 87.6100 (87.6100)  mean_accuracy: 73.5000 (73.5000)  mean_iou: 50.0200 (50.0200)
        kmeans: pixel_accuracy: 76.3200 (76.3200)  mean_accuracy: 56.8000 (56.8000)  mean_iou: 34.4500 (34.4500)
    
    pixel_accuracy: 86.0500 (86.0500)  mean_accuracy: 67.5700 (67.5700)  mean_iou: 46.6500 (46.6500)
    our_kmeans: pixel_accuracy: 88.2500 (88.2500)  mean_accuracy: 73.7300 (73.7300)  mean_iou: 52.7800 (52.7800)
    kmeans: pixel_accuracy: 75.4800 (75.4800)  mean_accuracy: 57.1000 (57.1000)  mean_iou: 34.2000 (34.2000)

    reco:
        mIoU (bi) 0.281 (0.282) | Pixel acc (bi) 0.833 (0.834)
        our_kmeans: mIoU (bi) 0.300 (0.300) | Pixel acc (bi) 0.846 (0.846)
        kmeans: mIoU (bi) 0.223 (0.224) | Pixel acc (bi) 0.770 (0.770)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/51 --dataset cityscapes --no-resume --backbone vit_base_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    pixel_accuracy: 84.7800 (84.7800)  mean_accuracy: 65.1200 (65.1200)  mean_iou: 44.1200 (44.1200)
    our_kmeans: pixel_accuracy: 87.9400 (87.9400)  mean_accuracy: 70.7200 (70.7200)  mean_iou: 49.5400 (49.5400)
    kmeans: pixel_accuracy: 72.7000 (72.7000)  mean_accuracy: 60.5600 (60.5600)  mean_iou: 34.9300 (34.9300)
    
    reco:
        mIoU (bi) 0.267 (0.268) | Pixel acc (bi) 0.814 (0.814)
        our_kmeans: mIoU (bi) 0.307 (0.307) | Pixel acc (bi) 0.842 (0.842)
        kmeans: mIoU (bi) 0.232 (0.232) | Pixel acc (bi) 0.748 (0.748)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/52 --dataset cityscapes --no-resume --backbone vit_large_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
    pixel_accuracy: 84.1300 (84.1300)  mean_accuracy: 65.4500 (65.4500)  mean_iou: 43.8000 (43.8000)
    our_kmeans: pixel_accuracy: 87.8200 (87.8200)  mean_accuracy: 68.9800 (68.9800)  mean_iou: 50.1700 (50.1700)
    kmeans:  pixel_accuracy: 62.2500 (62.2500)  mean_accuracy: 56.3800 (56.3800)  mean_iou: 31.6500 (31.6500)

    reco:
        mIoU (bi) 0.263 (0.263) | Pixel acc (bi) 0.802 (0.803)
        our_kmeans: mIoU (bi) 0.300 (0.300) | Pixel acc (bi) 0.842 (0.842)
        kmeans: mIoU (bi) 0.209 (0.209) | Pixel acc (bi) 0.663 (0.663)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/53 --dataset cityscapes --no-resume --backbone vit_base_patch8_224 --batch-size 1 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08



-------------------
ablation studies
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/30 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 128 --alpha 0.16 --reco --mode our --eval-only
    our: mIoU (bi) 0.327 (0.332) | Pixel acc (bi) 0.714 (0.716)
    mIoU (bi) 0.372 (0.372) | Pixel acc (bi) 0.739 (0.739)

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/31 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 64 --alpha 0.32 --reco --mode our --eval-only
    our: mIoU (bi) 0.275 (0.278) | Pixel acc (bi) 0.668 (0.671)
    mIoU (bi) 0.320 (0.320) | Pixel acc (bi) 0.730 (0.731)

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/32 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 512 --alpha 0.04 --reco --mode our --eval-only
    our: mIoU (bi) 0.375 (0.379) | Pixel acc (bi) 0.709 (0.711)
    mIoU (bi) 0.405 (0.405) | Pixel acc (bi) 0.736 (0.736)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/33 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.04 --reco --mode our --eval-only
    our: mIoU (bi) 0.393 (0.398) | Pixel acc (bi) 0.719 (0.721)
    mIoU (bi) 0.390 (0.390) | Pixel acc (bi) 0.742 (0.742)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/34 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.16 --reco --mode our --eval-only
    our: mIoU (bi) 0.394 (0.398) | Pixel acc (bi) 0.707 (0.708)
    mIoU (bi) 0.393 (0.393) | Pixel acc (bi) 0.736 (0.736)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/35 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --no_sg --reco --mode our --eval-only
    our: mIoU (bi) 0.368 (0.373) | Pixel acc (bi) 0.701 (0.702)
    mIoU (bi) 0.404 (0.404) | Pixel acc (bi) 0.739 (0.739)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/36 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --tau_max 1  --tau_min 1
    our: mIoU (bi) 0.368 (0.379) | Pixel acc (bi) 0.701 (0.705)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/37 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --tau_max 1  --tau_min 0.5
    our: mIoU (bi) 0.386 (0.393) | Pixel acc (bi) 0.721 (0.725)
    our (twice): mIoU (bi) 0.379 (0.386) | Pixel acc (bi) 0.706 (0.709)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/38 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --tau_max 0.5  --tau_min 0.3
    our: mIoU (bi) 0.347 (0.352) | Pixel acc (bi) 0.697 (0.700)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/39 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --num_nearestn_feature 32
    our: mIoU (bi) 0.362 (0.367) | Pixel acc (bi) 0.696 (0.699)
    mIoU (bi) 0.396 (0.396) | Pixel acc (bi) 0.732 (0.732)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/29 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --num_nearestn_feature 64
    our: mIoU (bi) 0.371 (0.377) | Pixel acc (bi) 0.686 (0.689)
    mIoU (bi) 0.423 (0.424) | Pixel acc (bi) 0.748 (0.748)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/28 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --num_nearestn_feature 128
    our: mIoU (bi) 0.383 (0.389) | Pixel acc (bi) 0.714 (0.717)
    mIoU (bi) 0.400 (0.401) | Pixel acc (bi) 0.741 (0.741)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/27 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --num_nearestn_feature 512
    our: mIoU (bi) 0.379 (0.383) | Pixel acc (bi) 0.707 (0.709)
    mIoU (bi) 0.402 (0.402) | Pixel acc (bi) 0.746 (0.746)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/21 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --num_nearestn_feature 64 --pixelwise_weight 0.1 --tau_max -1  --tau_min -1 --reco --mode our 
    our: mIoU (bi) 0.005 (0.006) | Pixel acc (bi) 0.237 (0.239)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/20 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --num_nearestn_feature 64 --pixelwise_weight 0.1 --reco --mode our --no-orthogonal_linear
    our: mIoU (bi) 0.270 (0.272) | Pixel acc (bi) 0.636 (0.638)

***** 

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/26 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --pixelwise_weight 0.
    our: mIoU (bi) 0.381 (0.386) | Pixel acc (bi) 0.712 (0.715)
    mIoU (bi) 0.387 (0.387) | Pixel acc (bi) 0.725 (0.725)
    CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/26-2 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --pixelwise_weight 0.
        our: mIoU (bi) 0.395 (0.400) | Pixel acc (bi) 0.715 (0.718)
        our_kmeans: mIoU (bi) 0.411 (0.412) | Pixel acc (bi) 0.742 (0.742)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/25 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --pixelwise_weight 0.1
    our: mIoU (bi) 0.374 (0.379) | Pixel acc (bi) 0.694 (0.697)
    mIoU (bi) 0.419 (0.420) | Pixel acc (bi) 0.739 (0.739)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/24 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --pixelwise_weight 0.5
    our: mIoU (bi) 0.384 (0.389) | Pixel acc (bi) 0.705 (0.707)
    mIoU (bi) 0.379 (0.380) | Pixel acc (bi) 0.728 (0.728)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/23 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco --mode our_kmeans --pixelwise_weight 0.7
    our: mIoU (bi) 0.376 (0.380) | Pixel acc (bi) 0.709 (0.711)
    mIoU (bi) 0.389 (0.389) | Pixel acc (bi) 0.740 (0.740)

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/19 --dataset pascal_context --no-resume --backbone vit_base_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0. --reco
    our: mIoU (bi) 0.359 (0.365) | Pixel acc (bi) 0.677 (0.681)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/18 --dataset pascal_context --no-resume --backbone vit_large_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0. --reco
    our: mIoU (bi) 0.283 (0.288) | Pixel acc (bi) 0.585 (0.588)

*** 

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/17 --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0. --reco
    our: mIoU (bi) 0.267 (0.268) | Pixel acc (bi) 0.815 (0.817)
    our_kmeans: mIoU (bi) 0.298 (0.298) | Pixel acc (bi) 0.828 (0.828)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/16 --dataset cityscapes --no-resume --backbone vit_base_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0. --reco
    our: mIoU (bi) 0.274 (0.275) | Pixel acc (bi) 0.816 (0.817)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/15 --dataset cityscapes --no-resume --backbone vit_large_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0. --reco
    our: mIoU (bi) 0.270 (0.272) | Pixel acc (bi) 0.785 (0.787)


CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/14 --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0.1 --reco 
    our: mIoU (bi) 0.266 (0.267) | Pixel acc (bi) 0.815 (0.817)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/13 --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0.5 --reco 
    our: mIoU (bi) 0.280 (0.281) | Pixel acc (bi) 0.826 (0.827)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/12 --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --pixelwise_weight 0.7 --reco
    our: mIoU (bi) 0.277 (0.279) | Pixel acc (bi) 0.822 (0.823)


zero-shot transfer
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/11 --dataset imagenet --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 -lr .001 --psi_k 256 --alpha 0.08 --reco
    city: mIoU (bi) 0.185 (0.185) | Pixel acc (bi) 0.810 (0.812)
    pascal_context: mIoU (bi) 0.151 (0.152) | Pixel acc (bi) 0.557 (0.558)

visualize eigenfunctions:
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/40-2 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --eval-only --vis_eigenfunc





-----------------for retuttal-----------------
CUDA_VISIBLE_DEVICES=1 python train.py --log-dir rebuttal_logs/0 --dataset pascal_context --no-resume --backbone vit_tiny_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco
    mIoU (bi) 0.316 (0.321) | Pixel acc (bi) 0.682 (0.687): : 11it [00:06,  1.57it/s]
CUDA_VISIBLE_DEVICES=2 python train.py --log-dir rebuttal_logs/1 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco
    mIoU (bi) 0.375 (0.379) | Pixel acc (bi) 0.697 (0.699): : 11it [00:05,  1.86it/s]
CUDA_VISIBLE_DEVICES=3 python train.py --log-dir rebuttal_logs/2 --dataset pascal_context --no-resume --backbone clip_ViT-B/16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco
    mIoU (bi) 0.367 (0.372) | Pixel acc (bi) 0.724 (0.726)
CUDA_VISIBLE_DEVICES=4 python train.py --log-dir rebuttal_logs/3_3 --dataset pascal_context --no-resume --backbone clip_ViT-L/14 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco  --tau_min 0.5
    mIoU (bi) 0.327 (0.340) | Pixel acc (bi) 0.682 (0.695)
    mIoU (bi) 0.375 (0.387) | Pixel acc (bi) 0.711 (0.722)
    mIoU (bi) 0.332 (0.346) | Pixel acc (bi) 0.694 (0.707)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir rebuttal_logs/4 --dataset pascal_context --no-resume --backbone clip_ViT-L/14@336px --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --reco  --tau_min 0.5
    mIoU (bi) 0.431 (0.440) | Pixel acc (bi) 0.773 (0.780)
CUDA_VISIBLE_DEVICES=6 python train.py --log-dir rebuttal_logs/5 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 60 --alpha 0.3413 --reco
    mIoU (bi) 0.277 (0.280) | Pixel acc (bi) 0.701 (0.704)
    CUDA_VISIBLE_DEVICES=6 python train.py --log-dir rebuttal_logs/5 --dataset pascal_context --resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 60 --alpha 0.3413 --reco --hungarian_match
        mIoU (bi) 0.232 (0.235) | Pixel acc (bi) 0.405 (0.408)
```