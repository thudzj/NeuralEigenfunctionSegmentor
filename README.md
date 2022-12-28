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
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/40 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    our: pixel_accuracy: 73.1700 (73.1700)  mean_accuracy: 64.5200 (64.5200)  mean_iou: 45.2600 (45.2600)
    our_kmeans: pixel_accuracy: 75.4100 (75.4100)  mean_accuracy: 60.3400 (60.3400)  mean_iou: 44.6300 (44.6300)
    kmeans: pixel_accuracy: 61.9900 (61.9900)  mean_accuracy: 54.9000 (54.9000)  mean_iou: 36.7900 (36.7900)

    reco:
        our: mIoU (bi) 0.367 (0.371) | Pixel acc (bi) 0.688 (0.689)
        our_kmeans: mIoU (bi) 0.395 (0.395) | Pixel acc (bi) 0.744 (0.744)
        kmeans: mIoU (bi) 0.289 (0.289) | Pixel acc (bi) 0.618 (0.619)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/41 --dataset pascal_context --no-resume --backbone vit_base_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 70.6500 (70.6500)  mean_accuracy: 64.6100 (64.6100)  mean_iou: 43.6900 (43.6900)
    our_kmeans: pixel_accuracy: 74.2900 (74.2900)  mean_accuracy: 59.3400 (59.3400)  mean_iou: 42.9300 (42.9300)
    kmeans: pixel_accuracy: 57.8200 (57.8200)  mean_accuracy: 52.7400 (52.7400)  mean_iou: 33.0200 (33.0200)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/42 --dataset pascal_context --no-resume --backbone vit_large_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 65.8200 (65.8200)  mean_accuracy: 62.1900 (62.1900)  mean_iou: 40.3500 (40.3500)
    our_kmeans: pixel_accuracy: 74.2500 (74.2500)  mean_accuracy: 54.2000 (54.2000)  mean_iou: 40.9500 (40.9500)
    kmeans: pixel_accuracy: 44.4600 (44.4600)  mean_accuracy: 43.3700 (43.3700)  mean_iou: 23.5200 (23.5200)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/43 --dataset pascal_context --no-resume --backbone vit_base_patch8_224 --batch-size 6 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 73.8400 (73.8400)  mean_accuracy: 56.6200 (56.6200)  mean_iou: 42.6400 (42.6400)

--------------------

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/44 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 63.2800 (63.2800)  mean_accuracy: 39.0100 (39.0100)  mean_iou: 21.5700 (21.5700)
    our_kmeans: pixel_accuracy: 62.4900 (62.4900)  mean_accuracy: 47.1600 (47.1600)  mean_iou: 23.5800 (23.5800)
    kmeans: pixel_accuracy: 50.7000 (50.7000)  mean_accuracy: 41.3700 (41.3700)  mean_iou: 19.2200 (19.2200)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/45 --dataset ade20k --no-resume --backbone vit_base_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 61.0000 (61.0000)  mean_accuracy: 45.9800 (45.9800)  mean_iou: 25.0000 (25.0000)
    our_kmeans: pixel_accuracy: 62.2000 (62.2000)  mean_accuracy: 45.9100 (45.9100)  mean_iou: 23.3600 (23.3600)
    kmeans: pixel_accuracy: 48.6000 (48.6000)  mean_accuracy: 45.9900 (45.9900)  mean_iou: 19.6000 (19.6000)  

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/46 --dataset ade20k --no-resume --backbone vit_large_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2 --eval-only --mode our_kmeans
    pixel_accuracy: 59.0300 (59.0300)  mean_accuracy: 44.7000 (44.7000)  mean_iou: 25.2500 (25.2500) 
    our_kmeans: pixel_accuracy: 61.1400 (61.1400)  mean_accuracy: 42.9000 (42.9000)  mean_iou: 22.5800 (22.5800)
    kmeans: pixel_accuracy: 38.3500 (38.3500)  mean_accuracy: 42.8100 (42.8100)  mean_iou: 17.8400 (17.8400)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/47 --dataset ade20k --no-resume --backbone vit_base_patch8_224 --batch-size 6 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 66.4500 (66.4500)  mean_accuracy: 24.9300 (24.9300)  mean_iou: 17.4800 (17.4800)
    our_kmeans: pixel_accuracy: 62.0500 (62.0500)  mean_accuracy: 45.0000 (45.0000)  mean_iou: 22.5200 (22.5200)

--------------------
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/50 --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
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

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/51 --dataset cityscapes --no-resume --backbone vit_base_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 84.7800 (84.7800)  mean_accuracy: 65.1200 (65.1200)  mean_iou: 44.1200 (44.1200)
    our_kmeans: pixel_accuracy: 87.9400 (87.9400)  mean_accuracy: 70.7200 (70.7200)  mean_iou: 49.5400 (49.5400)
    kmeans: pixel_accuracy: 72.7000 (72.7000)  mean_accuracy: 60.5600 (60.5600)  mean_iou: 34.9300 (34.9300)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/52 --dataset cityscapes --no-resume --backbone vit_large_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2
    pixel_accuracy: 84.1300 (84.1300)  mean_accuracy: 65.4500 (65.4500)  mean_iou: 43.8000 (43.8000)
    our_kmeans: pixel_accuracy: 87.8200 (87.8200)  mean_accuracy: 68.9800 (68.9800)  mean_iou: 50.1700 (50.1700)
    kmeans:  pixel_accuracy: 62.2500 (62.2500)  mean_accuracy: 56.3800 (56.3800)  mean_iou: 31.6500 (31.6500)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/53 --dataset cityscapes --no-resume --backbone vit_base_patch8_224 --batch-size 1 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08 --tau_max 1 --tau_min 0.3 --num_nearestn_feature 256 --pixelwise_weight 0.3 --num_nearestn_pixel1 10 --num_nearestn_pixel2 5 --t 0.01 --psi_num_blocks 2

```
