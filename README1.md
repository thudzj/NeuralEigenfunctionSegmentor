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

-----
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/21 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11110 --upsample_factor 2 --cache_size 30
    loss: 1.4982 (1.4263)
    150 Stats ['final']: pixel_accuracy: 58.7500 (58.7500)  mean_accuracy: 13.2400 (13.2400)  mean_iou: 8.3000 (8.3000)
    300 Stats ['final']: pixel_accuracy: 61.0100 (61.0100)  mean_accuracy: 15.7100 (15.7100)  mean_iou: 10.3600 (10.3600)
    500 Stats ['final']: pixel_accuracy: 62.0800 (62.0800)  mean_accuracy: 19.0500 (19.0500)  mean_iou: 12.9400 (12.9400)
    600 Stats ['final']: pixel_accuracy: 62.6200 (62.6200)  mean_accuracy: 18.6500 (18.6500)  mean_iou: 12.5100 (12.5100)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/43 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11111 --upsample_factor 2 --cache_size 30 --pixelwise_adj_weight 8
    loss: 1.2956 (1.3739)
    150: Stats ['final']: pixel_accuracy: 60.9700 (60.9700)  mean_accuracy: 14.6300 (14.6300)  mean_iou: 9.3800 (9.3800)
    300: Stats ['final']: pixel_accuracy: 62.8100 (62.8100)  mean_accuracy: 17.9000 (17.9000)  mean_iou: 12.1300 (12.1300)
    500: Stats ['final']: pixel_accuracy: 63.5700 (63.5700)  mean_accuracy: 20.2200 (20.2200)  mean_iou: 13.6900 (13.6900)
    600: Stats ['final']: pixel_accuracy: 64.5800 (64.5800)  mean_accuracy: 21.2100 (21.2100)  mean_iou: 14.7600 (14.7600)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/31 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11112 --upsample_factor 2 --cache_size 30 --psi_k 256
    loss: 1.3076 (1.3881)
    150+crf: Stats ['final']: pixel_accuracy: 59.4900 (59.4900)  mean_accuracy: 13.4800 (13.4800)  mean_iou: 8.6500 (8.6500)
    150: Stats ['final']: pixel_accuracy: 59.4600 (59.4600)  mean_accuracy: 12.9400 (12.9400)  mean_iou: 8.1100 (8.1100)
    300: Stats ['final']: pixel_accuracy: 61.0900 (61.0900)  mean_accuracy: 16.0500 (16.0500)  mean_iou: 10.5600 (10.5600)
    500: Stats ['final']: pixel_accuracy: 62.1800 (62.1800)  mean_accuracy: 17.6000 (17.6000)  mean_iou: 12.1600 (12.1600)
    600: Stats ['final']: pixel_accuracy: 62.6500 (62.6500)  mean_accuracy: 18.4800 (18.4800)  mean_iou: 12.7000 (12.7000) 

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/32 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11113 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8
    loss: 1.2443 (1.3469)
    150: Stats ['final']: pixel_accuracy: 62.1600 (62.1600)  mean_accuracy: 14.0700 (14.0700)  mean_iou: 9.2500 (9.2500)
    300: Stats ['final']: pixel_accuracy: 63.9000 (63.9000)  mean_accuracy: 17.5000 (17.5000)  mean_iou: 11.7100 (11.7100)
    500: Stats ['final']: pixel_accuracy: 64.8500 (64.8500)  mean_accuracy: 20.2000 (20.2000)  mean_iou: 13.5900 (13.5900) 
    600: Stats ['final']: pixel_accuracy: 64.7400 (64.7400)  mean_accuracy: 20.4300 (20.4300)  mean_iou: 14.0100 (14.0100)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/33 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11114 --upsample_factor 2 --cache_size 30 --psi_k 512
    loss: 1.2966 (1.3760) 
    150: Stats ['final']: pixel_accuracy: 58.7400 (58.7400)  mean_accuracy: 11.6500 (11.6500)  mean_iou: 7.7500 (7.7500)
    300: Stats ['final']: pixel_accuracy: 60.1300 (60.1300)  mean_accuracy: 14.9800 (14.9800)  mean_iou: 9.6500 (9.6500)
    500: Stats ['final']: pixel_accuracy: 61.5100 (61.5100)  mean_accuracy: 16.2200 (16.2200)  mean_iou: 10.8500 (10.8500)
    600: Stats ['final']: pixel_accuracy: 62.0800 (62.0800)  mean_accuracy: 16.6000 (16.6000)  mean_iou: 11.4300 (11.4300)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/34 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11135 --upsample_factor 2 --cache_size 30 --psi_k 512 --pixelwise_adj_weight 8
    loss: 1.3363 (1.3074)
    150: Stats ['final']: pixel_accuracy: 60.1300 (60.1300)  mean_accuracy: 12.4300 (12.4300)  mean_iou: 8.0000 (8.0000)
    300: Stats ['final']: pixel_accuracy: 63.3800 (63.3800)  mean_accuracy: 16.1300 (16.1300)  mean_iou: 10.8800 (10.8800)
    500: Stats ['final']: pixel_accuracy: 64.0700 (64.0700)  mean_accuracy: 16.9600 (16.9600)  mean_iou: 11.2500 (11.2500)
    600: Stats ['final']: pixel_accuracy: 64.6800 (64.6800)  mean_accuracy: 18.2800 (18.2800)  mean_iou: 12.4500 (12.4500)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/56 --dataset ade20k --resume --batch-size 10 --epochs 20 --input_l2_normalize --port 11116 --upsample_factor 2 --cache_size 30 --psi_k 256 --pixelwise_adj_weight 8 --psi_res --psi_num_layers 8 --psi_act_type relu --optimizer our_lars -lr 0.3 --alpha 1
    loss: 1.3055 (1.4520)
    150: Stats ['final']: pixel_accuracy: 58.9200 (58.9200)  mean_accuracy: 13.3000 (13.3000)  mean_iou: 8.4900 (8.4900)
    300: Stats ['final']: pixel_accuracy: 61.3000 (61.3000)  mean_accuracy: 16.1500 (16.1500)  mean_iou: 10.4500 (10.4500)
    500: Stats ['final']: pixel_accuracy: 62.7800 (62.7800)  mean_accuracy: 17.7000 (17.7000)  mean_iou: 11.9200 (11.9200) 
    600: Stats ['final']: pixel_accuracy: 62.7600 (62.7600)  mean_accuracy: 18.1800 (18.1800)  mean_iou: 12.4600 (12.4600)

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/baseline --dataset ade20k --no-resume --batch-size 10 --epochs 0 --port 11137 --psi_num_layers 0
    150: Stats ['final']: pixel_accuracy: 56.7500 (56.7500)  mean_accuracy: 17.5000 (17.5000)  mean_iou: 10.2400 (10.2400)
    300: Stats ['final']: pixel_accuracy: 59.3200 (59.3200)  mean_accuracy: 21.6000 (21.6000)  mean_iou: 13.5900 (13.5900)
    500: Stats ['final']: pixel_accuracy: 60.8900 (60.8900)  mean_accuracy: 25.9300 (25.9300)  mean_iou: 16.9000 (16.9000)
    600: Stats ['final']: pixel_accuracy: 61.7100 (61.7100)  mean_accuracy: 25.0800 (25.0800)  mean_iou: 17.4400 (17.4400)

---
CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/baseline --dataset ade20k --no-resume --batch-size 16 --epochs 20 --port 11137 --psi_num_layers 0 --optimizer our_lars -lr .3 --kmeans_n_cls 600 
    loss: 0.9747 (1.0018)
    Stats ['final']: pixel_accuracy: 61.1600 (61.1600)  mean_accuracy: 27.1300 (27.1300)  mean_iou: 17.9300 (17.9300)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/64 --port 11114 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 512 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --kmeans_n_cls 600
    loss: 1.0520 (1.1218)  loss2: 1.2741 (1.3331)




CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/60 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --kmeans_n_cls 600
    loss: 1.0623 (1.0999)  loss2: 1.2774 (1.3380)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/61 --port 11111 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 512 --optimizer adamw -lr .0001 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --kmeans_n_cls 600
    loss: 1.1463 (1.1187)  loss2: 1.4851 (1.4438)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/62 --port 11112 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 20 --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --kmeans_n_cls 600
    loss: 1.0420 (1.0992)  loss2: 1.3086 (1.3506)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/63 --port 11113 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 50 --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --kmeans_n_cls 600
    loss: 1.0385 (1.1051)  loss2: 1.2804 (1.3670)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/65 --port 11114 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --psi_num_layers 2 --kmeans_n_cls 600
    loss: 0.9010 (0.9542)  loss2: 1.2252 (1.2667)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/66 --port 11115 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --psi_num_layers 6 --kmeans_n_cls 600
    loss: 1.1243 (1.1518)  loss2: 1.3547 (1.3698)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/67 --port 11116 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 256 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 2 --kmeans_n_cls 600
    loss: 1.0185 (1.1021)  loss2: 1.2207 (1.3761)

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/68 --port 11117 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 1024 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.5 --kmeans_n_cls 600
    loss: 1.0838 (1.0992)  loss2: 1.2880 (1.2826)

---------------
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/70 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --kmeans_n_cls 600
    three: loss: 0.9863 (0.9901)  loss2: 1.4355 (1.3484)
    two layer projector: loss: 1.0472 (1.0013)  loss2: 1.2741 (1.2949)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/71 --port 11111 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 256 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 2 --kmeans_n_cls 600
    loss: 0.9812 (0.9926)  loss2: 1.3601 (1.4133)
    loss: 0.9409 (1.0001)  loss2: 1.3426 (1.3508)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/72 --port 11112 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 1024 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.5 --kmeans_n_cls 600
    loss: 1.0290 (0.9943)  loss2: 1.2827 (1.2930)
    loss: 0.9379 (0.9967)  loss2: 1.2209 (1.2362)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/73 --port 11113 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kmeans_n_cls 600
    *
    loss: 0.9641 (0.9964)  loss2: 1.1657 (1.1985)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/74 --port 11114 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 512 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 1 --psi_num_layers 6 --kmeans_n_cls 600
    loss: 0.9388 (1.0169)  loss2: 1.3263 (1.3690)
    loss: 1.0117 (1.0125)  loss2: 1.3039 (1.3191)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/75 --port 11115 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 256 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 2 --psi_num_layers 6 --kmeans_n_cls 600
    loss: 0.9582 (1.0210)  loss2: 1.3122 (1.4453)
     loss: 1.0412 (1.0250)  loss2: 1.4258 (1.3903)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/76 --port 11116 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 1024 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.5 --psi_num_layers 6 --kmeans_n_cls 600
    loss: 0.9137 (1.0166)  loss2: 1.1889 (1.3092)
    loss: 0.9892 (1.0129)  loss2: 1.2404 (1.2654)

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/77 --port 11117 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --psi_num_layers 6 --kmeans_n_cls 600
    *
    loss: 0.9525 (1.0178)  loss2: 1.1298 (1.2208)

---------------
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/80 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn8_innerp' --psi_num_layers 2 --kmeans_n_cls 600
    

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/80 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn8_innerp' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 1.0787 (1.0278)  loss2: 1.2053 (1.1465)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/81 --port 11111 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn16_innerp' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 1.0427 (1.0010)  loss2: 1.1597 (1.1156)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/82 --port 11112 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn32_innerp' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9323 (0.9869)  loss2: 1.0604 (1.1165) 

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/83 --port 11113 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_innerp' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9170 (0.9796)  loss2: 1.0426 (1.1013)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/84 --port 11114 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn8_l2' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9588 (1.0103)  loss2: 1.0938 (1.1563)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/85 --port 11115 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn16_l2' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9041 (0.9972)  loss2: 1.0241 (1.1122)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/86 --port 11116 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn32_l2' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9879 (0.9850)  loss2: 1.0910 (1.1003)

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/87 --port 11117 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_l2' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9916 (0.9758)  loss2: 1.1183 (1.1003)

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/90 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_l2' --psi_num_layers 2 --kmeans_n_cls 600
    loss: 0.8728 (0.9607)  loss2: 0.9590 (1.0586)

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/90 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn128_innerp' --psi_num_layers 4 --kmeans_n_cls 600
     loss: 0.9713 (0.9786)  loss2: 1.1281 (1.1189)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/91 --port 11111 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn256_innerp' --psi_num_layers 4 --kmeans_n_cls 600
     loss: 1.0242 (0.9763)  loss2: 1.2145 (1.1693)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/92 --port 11112 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn128_l2' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9434 (0.9758)  loss2: 1.0640 (1.1136)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/93 --port 11113 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn256_l2' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9902 (0.9826)  loss2: 1.1623 (1.1737)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/94 --port 11114 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn32_cosine' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9388 (0.9856)  loss2: 1.0379 (1.1026)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/95 --port 11115 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_cosine' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 0.9301 (0.9767)  loss2: 1.0559 (1.0985)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/96 --port 11116 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn128_cosine' --psi_num_layers 4 --kmeans_n_cls 600
    loss: 1.0226 (0.9730)  loss2: 1.1756 (1.1103)

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/97 --port 11117 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_res --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn256_cosine' --psi_num_layers 4 --kmeans_n_cls 600
     loss: 0.9756 (0.9818)  loss2: 1.1401 (1.1744)


----------

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/98 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --psi_num_layers 2 --kmeans_n_cls 600
    loss: 0.9670 (0.9571)  loss2: 1.1756 (1.1862)
    Stats ['final']: pixel_accuracy: 62.8900 (62.8900)  mean_accuracy: 25.0900 (25.0900)  mean_iou: 16.8200 (16.8200)

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/91 --port 11111 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --psi_num_layers 3 --kmeans_n_cls 600
    loss: 1.0013 (1.0469)  loss2: 1.1560 (1.2177)
    Stats ['final']: pixel_accuracy: 61.5900 (61.5900)  mean_accuracy: 22.9200 (22.9200)  mean_iou: 15.6000 (15.6000)

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/92 --port 11112 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_l2' --psi_num_layers 2 --kmeans_n_cls 600
    loss: 0.8348 (0.9349)  loss2: 0.9340 (1.0642)
    Stats ['final']: pixel_accuracy: 65.3600 (65.3600)  mean_accuracy: 27.1400 (27.1400)  mean_iou: 18.2400 (18.2400)

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/93 --port 11113 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_l2' --psi_num_layers 3 --kmeans_n_cls 600
    loss: 0.8993 (0.9655)  loss2: 1.0351 (1.0905)
    Stats ['final']: pixel_accuracy: 66.0300 (66.0300)  mean_accuracy: 24.1200 (24.1200)  mean_iou: 16.4100 (16.4100)

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/94 --port 11114 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel '40_ball' --psi_num_layers 2 --kmeans_n_cls 600
    loss: 1.0153 (0.9561)  loss2: 1.2469 (1.1550)
    Stats ['final']: pixel_accuracy: 63.9400 (63.9400)  mean_accuracy: 27.7800 (27.7800)  mean_iou: 18.3900 (18.3900)

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/95 --port 11115 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel '40_ball' --psi_num_layers 3 --kmeans_n_cls 600
    loss: 0.9802 (1.0187)  loss2: 1.1272 (1.1764)
    Stats ['final']: pixel_accuracy: 62.7100 (62.7100)  mean_accuracy: 25.2100 (25.2100)  mean_iou: 16.8000 (16.8000)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/96 --port 11116 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel '45_ball' --psi_num_layers 2 --kmeans_n_cls 600
    loss: 0.9717 (0.9600)  loss2: 1.2937 (1.2342)
    Stats ['final']: pixel_accuracy: 62.2100 (62.2100)  mean_accuracy: 26.8900 (26.8900)  mean_iou: 18.0000 (18.0000)

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/97 --port 11117 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel '45_ball' --psi_num_layers 3 --kmeans_n_cls 600
    loss: 1.0113 (1.0498)  loss2: 1.2767 (1.2798)
    Stats ['final']: pixel_accuracy: 59.4000 (59.4000)  mean_accuracy: 22.2100 (22.2100)  mean_iou: 15.1500 (15.1500)

------
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/98 --port 11110 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .0003 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_l2' --psi_num_layers 2 --kmeans_n_cls 600
     loss: 0.8823 (0.9361)  loss2: 1.0831 (1.0644)
     Stats ['final']: pixel_accuracy: 65.5400 (65.5400)  mean_accuracy: 27.2000 (27.2000)  mean_iou: 18.1300 (18.1300)

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/96 --port 11116 --dataset ade20k --no-resume --batch-size 16 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --t_for_neuralef 1 --alpha 0.25 --kernel 'nearestn64_l2' --psi_num_layers 2 --kmeans_n_cls 600
    loss: 0.8858 (0.9124)  loss2: 1.0260 (1.0637)
    Stats ['final']: pixel_accuracy: 65.0800 (65.0800)  mean_accuracy: 27.5300 (27.5300)  mean_iou: 18.7200 (18.7200)
-------

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/99 --port 11110 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --alpha 0.25 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs  --kernel 'nearestn128_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/91 --port 11111 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --alpha 0.25 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn256_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/92 --port 11112 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --alpha 0.25 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn512_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/93 --port 11113 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 2048 --alpha 0.25 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn1024_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/94 --port 11114 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_num_features 4096 --psi_k 2048 --alpha 0.25 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn128_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/95 --port 11115 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_num_features 4096 --psi_k 4096 --alpha 0.125 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn128_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/96 --port 11116 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_num_features 4096 --psi_k 4096 --alpha 0.125 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn128_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600
    4096-4096-4096

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/97 --port 11117 --dataset ade20k --no-resume --amp --batch-size 10 --epochs 20 --psi_act_type 'relu' --psi_norm_type 'bn' --cache_size 0 --psi_k 4096 --alpha 0.125 --optimizer adamw -lr .001 --psi_projector --kmeans_use_hidden_outputs --kernel 'nearestn128_innerp' --psi_num_layers 2 --upsample_factor 2 --kmeans_n_cls 600

--------

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/95 --port 11115 --dataset ade20k --no-resume --batch-size 14 --epochs 5 --cache_size 0 --psi_num_features 512 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn64_l2' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600 
    # 24 heads 32
    loss: 0.9072 (0.9292)  loss2: 0.9234 (0.9381)  loss3: 0.9422 (0.9522)

CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/90 --port 11110 --dataset ade20k --no-resume --batch-size 13 --epochs 5 --cache_size 0 --psi_num_features 512 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn64_l2' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600
    loss: 0.8037 (0.9344)  loss2: 0.8208 (0.9434)  loss3: 0.8409 (0.9533) 
    Stats ['final']: pixel_accuracy: 73.8500 (73.8500)  mean_accuracy: 29.1700 (29.1700)  mean_iou: 21.2500 (21.2500)

---------
CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/90 --port 11110 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 12 --epochs 5 --cache_size 0 --psi_num_features 768 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn64_cosine_20_10' --psi_num_layers 3 --upsample_factor 1 --kmeans_n_cls 600 --pixelwise_adj_weight 8 




CUDA_VISIBLE_DEVICES=0 python train.py --log-dir logs/90 --port 11110 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_10_10' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/91 --port 11111 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_0_10' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=2 python train.py --log-dir logs/92 --port 11112 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_10_0' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=3 python train.py --log-dir logs/93 --port 11113 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_20_0' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=4 python train.py --log-dir logs/94 --port 11114 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_20_10' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=5 python train.py --log-dir logs/95 --port 11115 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_10_5' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=6 python train.py --log-dir logs/96 --port 11116 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_0_5' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600

CUDA_VISIBLE_DEVICES=7 python train.py --log-dir logs/97 --port 11117 --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 5 --cache_size 0 --psi_num_features 576 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn32_cosine_1_0.5_5_0' --psi_num_layers 4 --upsample_factor 1 --kmeans_n_cls 600




# inner-image kernel
# e-ball


CUDA_VISIBLE_DEVICES=1 python train.py --log-dir logs/91 --port 11111 --dataset ade20k --no-resume --backbone vit_base_patch16_224_dino --batch-size 8 --epochs 5 --cache_size 0 --psi_num_features 768 --psi_k 2048 --alpha 0.02 --optimizer adam -lr .001 --psi_transformer --kernel 'nearestn16_innerp_0_0' --psi_num_layers 3 --upsample_factor 2 --kmeans_n_cls 600 --pixelwise_adj_weight 8

l2 normalize
one layer projector


```

and test with
```
CUDA_VISIBLE_DEVICES=1 python inference.py --model-path logs/normalized_adjacency-l2/checkpoint.pth -i $DATASET/ade20k/ADEChallengeData2016/images/validation -o segmaps/ 
```