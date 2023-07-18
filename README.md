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
python train.py --log-dir logs/p --dataset pascal_context --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08

python train.py --log-dir logs/a --dataset ade20k --no-resume --backbone vit_small_patch16_384 --batch-size 16 --epochs 20 -lr .001 --psi_k 512 --alpha 0.04 

python train.py --log-dir logs/c --dataset cityscapes --no-resume --backbone vit_small_patch16_384 --batch-size 8 --epochs 40 -lr .001 --psi_k 256 --alpha 0.08
```