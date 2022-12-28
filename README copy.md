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
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/30 --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'    pixel_accuracy: 64.6800 (64.6800)  mean_accuracy: 48.5500 (48.5500)  mean_iou: 37.1200 (37.1200)
    'original_feature'           pixel_accuracy: 62.7800 (62.7800)  mean_accuracy: 47.4300 (47.4300)  mean_iou: 35.7900 (35.7900)
    'original_all_feature'       pixel_accuracy: 63.0600 (63.0600)  mean_accuracy: 46.4700 (46.4700)  mean_iou: 34.9000 (34.9000)
    'hidden_feature'             pixel_accuracy: 61.3700 (61.3700)  mean_accuracy: 42.1700 (42.1700)  mean_iou: 32.1200 (32.1200)
    CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/30-lp --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 64 --epochs 40 -lr .001 --kmeans_feature 'original&hidden_feature' --linear_probe_given logs/30/checkpoint.pth
                                 pixel_accuracy: 75.5300 (75.5300)  mean_accuracy: 59.8700 (59.8700)  mean_iou: 49.7100 (49.7100)
    CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/30-lp --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 64 --epochs 40 -lr .001 --kmeans_feature 'original_feature' --linear_probe_given yes
                                 pixel_accuracy: 73.1000 (73.1000)  mean_accuracy: 58.3400 (58.3400)  mean_iou: 47.7000 (47.7000)
    CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/30-lp --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 64 --epochs 40 -lr .001 --kmeans_feature 'original_all_feature' --linear_probe_given yes
                                 pixel_accuracy: 75.0800 (75.0800)  mean_accuracy: 59.1800 (59.1800)  mean_iou: 49.2600 (49.2600)
    CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/30-lp --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 64 --epochs 40 -lr .001 --kmeans_feature 'hidden_feature' --linear_probe_given logs/30/checkpoint.pth
                                 pixel_accuracy: 74.8700 (74.8700)  mean_accuracy: 56.4300 (56.4300)  mean_iou: 46.7800 (46.7800)


CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/31 --dataset pascal_context --no-resume --backbone vit_base_patch16_384 --batch-size 16 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'    pixel_accuracy: 60.2600 (60.2600)  mean_accuracy: 46.6300 (46.6300)  mean_iou: 34.2300 (34.2300)
    'original_feature'           pixel_accuracy: 59.9300 (59.9300)  mean_accuracy: 46.0400 (46.0400)  mean_iou: 33.7400 (33.7400)
    'original_all_feature'       pixel_accuracy: 59.1900 (59.1900)  mean_accuracy: 42.6600 (42.6600)  mean_iou: 31.1800 (31.1800)
    'hidden_feature'             pixel_accuracy: 58.6600 (58.6600)  mean_accuracy: 41.6200 (41.6200)  mean_iou: 32.1800 (32.1800)
    'original&hidden_feature'+lp pixel_accuracy: 76.4600 (76.4600)  mean_accuracy: 61.6900 (61.6900)  mean_iou: 51.5600 (51.5600)
    'original_feature'+lp        pixel_accuracy: 74.5900 (74.5900)  mean_accuracy: 60.9200 (60.9200)  mean_iou: 50.2100 (50.2100)
    'original_all_feature'+lp    pixel_accuracy: 76.3600 (76.3600)  mean_accuracy: 61.1900 (61.1900)  mean_iou: 51.4000 (51.4000)
    'hidden_feature'+lp          pixel_accuracy: 76.2400 (76.2400)  mean_accuracy: 59.3600 (59.3600)  mean_iou: 49.7000 (49.7000)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/32 --dataset pascal_context --no-resume --backbone vit_large_patch16_384 --batch-size 16 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'    pixel_accuracy: 48.8400 (48.8400)  mean_accuracy: 35.4300 (35.4300)  mean_iou: 25.0900 (25.0900)
    'original_feature'           pixel_accuracy: 48.4400 (48.4400)  mean_accuracy: 37.1400 (37.1400)  mean_iou: 24.7300 (24.7300)
    'original_all_feature'       pixel_accuracy: 47.3500 (47.3500)  mean_accuracy: 34.1600 (34.1600)  mean_iou: 23.5000 (23.5000)
    'hidden_feature'             pixel_accuracy: 49.2000 (49.2000)  mean_accuracy: 36.3600 (36.3600)  mean_iou: 27.0200 (27.0200)
    'original&hidden_feature'+lp pixel_accuracy: 75.1800 (75.1800)  mean_accuracy: 60.6800 (60.6800)  mean_iou: 50.5900 (50.5900)
    'original_feature'+lp        pixel_accuracy: 71.4600 (71.4600)  mean_accuracy: 57.8500 (57.8500)  mean_iou: 47.4200 (47.4200)
    'original_all_feature'+lp    pixel_accuracy: 75.1500 (75.1500)  mean_accuracy: 59.6700 (59.6700)  mean_iou: 50.3900 (50.3900)
    'hidden_feature'+lp          pixel_accuracy: 74.8100 (74.8100)  mean_accuracy: 59.1000 (59.1000)  mean_iou: 48.8900 (48.8900)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/33 --dataset pascal_context --no-resume --backbone vit_base_patch8_224 --batch-size 6 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'    pixel_accuracy: 54.5500 (54.5500)  mean_accuracy: 37.2000 (37.2000)  mean_iou: 26.3600 (26.3600)
    'original_feature'           pixel_accuracy: 54.7000 (54.7000)  mean_accuracy: 37.7700 (37.7700)  mean_iou: 26.2100 (26.2100)
    'original_all_feature'       pixel_accuracy: 54.3700 (54.3700)  mean_accuracy: 37.7300 (37.7300)  mean_iou: 26.9700 (26.9700)
    'hidden_feature'             pixel_accuracy: 43.6100 (43.6100)  mean_accuracy: 20.0600 (20.0600)  mean_iou: 14.6000 (14.6000)
    'original&hidden_feature'+lp pixel_accuracy: 75.0700 (75.0700)  mean_accuracy: 59.5500 (59.5500)  mean_iou: 49.3700 (49.3700)
    'original_feature'+lp        pixel_accuracy: 71.8100 (71.8100)  mean_accuracy: 57.3200 (57.3200)  mean_iou: 46.5700 (46.5700)
    'original_all_feature'+lp    pixel_accuracy: 74.6600 (74.6600)  mean_accuracy: 59.1000 (59.1000)  mean_iou: 49.1200 (49.1200)
    'hidden_feature'+lp          pixel_accuracy: 75.6200 (75.6200)  mean_accuracy: 58.8500 (58.8500)  mean_iou: 49.0200 (49.0200)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/20 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --kmeans_n_cls 600 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'     pixel_accuracy: 62.7800 (62.7800)  mean_accuracy: 25.6500 (25.6500)  mean_iou: 17.1200 (17.1200)
    'original_feature'            pixel_accuracy: 61.6800 (61.6800)  mean_accuracy: 26.6200 (26.6200)  mean_iou: 17.7300 (17.7300)
    'original_all_feature'        pixel_accuracy: 62.7000 (62.7000)  mean_accuracy: 27.7500 (27.7500)  mean_iou: 18.6200 (18.6200)
    'hidden_feature'              pixel_accuracy: 62.7300 (62.7300)  mean_accuracy: 17.0600 (17.0600)  mean_iou: 11.1400 (11.1400)
    CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/20-lp --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --kmeans_feature 'original&hidden_feature' --linear_probe_given logs/20/checkpoint.pth
                                  pixel_accuracy: 76.7900 (76.7900)  mean_accuracy: 49.1600 (49.1600)  mean_iou: 38.1800 (38.1800)
    CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/20-lp --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --kmeans_feature 'original_feature' --linear_probe_given yes
                                  pixel_accuracy: 74.2200 (74.2200)  mean_accuracy: 47.3100 (47.3100)  mean_iou: 36.4100 (36.4100)
    CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/20-lp --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --kmeans_feature 'original_all_feature' --linear_probe_given yes
                                  pixel_accuracy: 76.6100 (76.6100)  mean_accuracy: 49.6500 (49.6500)  mean_iou: 38.4900 (38.4900)
    CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/20-lp --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --kmeans_feature 'hidden_feature' --linear_probe_given logs/20/checkpoint.pth
                                  pixel_accuracy: 75.3500 (75.3500)  mean_accuracy: 42.2300 (42.2300)  mean_iou: 32.4600 (32.4600)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/21 --dataset ade20k --no-resume --backbone vit_base_patch16_384 --batch-size 14 --epochs 20 -lr .001 --kmeans_n_cls 600 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'     pixel_accuracy: 61.9400 (61.9400)  mean_accuracy: 31.7500 (31.7500)  mean_iou: 20.8200 (20.8200)
    'original_feature'            pixel_accuracy: 61.4100 (61.4100)  mean_accuracy: 32.0300 (32.0300)  mean_iou: 21.0500 (21.0500)
    'original_all_feature'        pixel_accuracy: 61.0500 (61.0500)  mean_accuracy: 31.6400 (31.6400)  mean_iou: 20.4300 (20.4300)
    'hidden_feature'              pixel_accuracy: 61.7600 (61.7600)  mean_accuracy: 17.4200 (17.4200)  mean_iou: 12.3500 (12.3500)
    'original&hidden_feature'+lp  pixel_accuracy: 78.5600 (78.5600)  mean_accuracy: 53.7600 (53.7600)  mean_iou: 42.1200 (42.1200)
    'original_feature'+lp         pixel_accuracy: 76.3900 (76.3900)  mean_accuracy: 52.1600 (52.1600)  mean_iou: 40.7700 (40.7700)
    'original_all_feature'+lp     pixel_accuracy: 78.3500 (78.3500)  mean_accuracy: 53.1500 (53.1500)  mean_iou: 42.1400 (42.1400)
    'hidden_feature'+lp           pixel_accuracy: 77.9000 (77.9000)  mean_accuracy: 50.0300 (50.0300)  mean_iou: 39.7200 (39.7200)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/22 --dataset ade20k --no-resume --backbone vit_large_patch16_384 --batch-size 14 --epochs 20 -lr .001 --kmeans_n_cls 600 --kmeans_feature 'original&hidden_feature' --eval-only
    'original&hidden_feature'     pixel_accuracy: 52.2200 (52.2200)  mean_accuracy: 26.7300 (26.7300)  mean_iou: 17.2800 (17.2800)
    'original_feature'            pixel_accuracy: 50.8700 (50.8700)  mean_accuracy: 29.0100 (29.0100)  mean_iou: 18.1400 (18.1400)
    'original_all_feature'        pixel_accuracy: 51.6200 (51.6200)  mean_accuracy: 27.1200 (27.1200)  mean_iou: 17.4800 (17.4800)
    'hidden_feature'              pixel_accuracy: 53.6900 (53.6900)  mean_accuracy: 17.9700 (17.9700)  mean_iou: 11.6700 (11.6700)
    'original&hidden_feature'+lp  pixel_accuracy: 78.1100 (78.1100)  mean_accuracy: 52.9600 (52.9600)  mean_iou: 42.0400 (42.0400)
    'original_feature'+lp         pixel_accuracy: 74.1800 (74.1800)  mean_accuracy: 50.6500 (50.6500)  mean_iou: 39.8700 (39.8700)
    'original_all_feature'+lp     pixel_accuracy: 77.8800 (77.8800)  mean_accuracy: 52.6200 (52.6200)  mean_iou: 41.9800 (41.9800)
    'hidden_feature'+lp           pixel_accuracy: 77.4500 (77.4500)  mean_accuracy: 51.1000 (51.1000)  mean_iou: 40.5500 (40.5500)


CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/23 --dataset ade20k --no-resume --backbone vit_base_patch8_224 --batch-size 6 --epochs 20 -lr .001 --kmeans_n_cls 600 --kmeans_feature 'original&hidden_feature' --psi_num_blocks 2 --eval-only
    'original&hidden_feature'     pixel_accuracy: 57.1900 (57.1900)  mean_accuracy: 26.9600 (26.9600)  mean_iou: 17.7900 (17.7900)
```


```
dino exps

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/10 --dataset pascal_context --no-resume --backbone vit_small_patch16_224_dino --batch-size 16 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' 
    'original&hidden_feature'     pixel_accuracy: 51.8300 (51.8300)  mean_accuracy: 24.1300 (24.1300)  mean_iou: 16.9600 (16.9600)
    'original_feature'            pixel_accuracy: 51.9900 (51.9900)  mean_accuracy: 24.4200 (24.4200)  mean_iou: 17.7800 (17.7800)
    'original_all_feature'        pixel_accuracy: 52.1800 (52.1800)  mean_accuracy: 25.0100 (25.0100)  mean_iou: 18.2300 (18.2300)
    'hidden_feature'              pixel_accuracy: 49.8300 (49.8300)  mean_accuracy: 23.5900 (23.5900)  mean_iou: 16.3500 (16.3500)
    'original&hidden_feature'+lp  
    'original_feature'+lp         
    'original_all_feature'+lp     
    'hidden_feature'+lp           

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/11 --dataset pascal_context --no-resume --backbone vit_base_patch16_224_dino --batch-size 16 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature'
    'original&hidden_feature'     pixel_accuracy: 55.4200 (55.4200)  mean_accuracy: 29.3700 (29.3700)  mean_iou: 22.2400 (22.2400)
    'original_feature'            
    'original_all_feature'        
    'hidden_feature'              
    'original&hidden_feature'+lp  
    'original_feature'+lp         
    'original_all_feature'+lp     
    'hidden_feature'+lp           

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/12 --dataset pascal_context --no-resume --backbone vit_small_patch8_224_dino --batch-size 8 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' 
    'original&hidden_feature'     
    'original_feature'            
    'original_all_feature'        
    'hidden_feature'              
    'original&hidden_feature'+lp  
    'original_feature'+lp         
    'original_all_feature'+lp     
    'hidden_feature'+lp           

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/13 --dataset pascal_context --no-resume --backbone vit_base_patch8_224_dino --batch-size 7 --epochs 40 -lr .001 --kmeans_n_cls 512 --kmeans_feature 'original&hidden_feature' 
    'original&hidden_feature'     
    'original_feature'            
    'original_all_feature'        
    'hidden_feature'              
    'original&hidden_feature'+lp  
    'original_feature'+lp         
    'original_all_feature'+lp     
    'hidden_feature'+lp           

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/14 --dataset ade20k --no-resume --backbone vit_small_patch16_224_dino --batch-size 16 --epochs 20 -lr .001 --kmeans_n_cls 600 --kmeans_feature 'original&hidden_feature' 
    'original&hidden_feature'     
    'original_feature'            
    'original_all_feature'        
    'hidden_feature'              
    'original&hidden_feature'+lp  
    'original_feature'+lp         
    'original_all_feature'+lp     
    'hidden_feature'+lp           

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/15 --dataset ade20k --no-resume --backbone vit_base_patch16_224_dino --batch-size 16 --epochs 20 -lr .001 --kmeans_n_cls 600 --kmeans_feature 'original&hidden_feature'
    'original&hidden_feature'     
    'original_feature'            
    'original_all_feature'        
    'hidden_feature'              
    'original&hidden_feature'+lp  
    'original_feature'+lp         
    'original_all_feature'+lp     
    'hidden_feature'+lp           

```

and test with
```
CUDA_VISIBLE_DEVICES=0 python -m eval.miou logs/30/checkpoint_final.pth pascal_context --singlescale
CUDA_VISIBLE_DEVICES=1 python -m eval.miou logs/31/checkpoint_final.pth pascal_context --singlescale
CUDA_VISIBLE_DEVICES=2 python -m eval.miou logs/32/checkpoint_final.pth pascal_context --singlescale
CUDA_VISIBLE_DEVICES=3 python -m eval.miou logs/33/checkpoint_final.pth pascal_context --singlescale
CUDA_VISIBLE_DEVICES=4 python -m eval.miou logs/20/checkpoint_final.pth ade20k --singlescale
CUDA_VISIBLE_DEVICES=5 python -m eval.miou logs/21/checkpoint_final.pth ade20k --singlescale
```
