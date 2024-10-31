# LSRCAN
Laser Stripes Residual Channel Attention Network


## Environment
*  pytorch >=1.0
* python 3.6
* numpy



## Train
* dataset
* prepare


Like [IMDN](https://github.com/Zheng222/IMDN), convert png files to npy files:
  ```python
  python scripts/png2npy.py --pathFrom /path/to/DIV2K/ --pathTo /path/to/DIV2K_decoded/
  ```
* Training
```shell
python train.py --scale 2 --patch_size 96

```

## Test
Example:
* test B100 X4
```shell
python test.py --is_y --test_hr_folder Set5all/Set5_LR/X1 --test_lr_folder Set5all/Set5_LR/X4  --output_folder RE/sed/x4 --checkpoint experiment/sed/checkpoint__sed_x4/epoch_980.pth --upscale_factor 4


```
