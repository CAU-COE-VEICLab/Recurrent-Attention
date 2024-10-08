# Swin-Unet-RecA
**Swin-Unet-RecA is to replace all the Self-Attention in Swin-Unet's encoder with the Recurrent-Attention-H. Note that we only replace Self-Attention and keep the design of Swin-Transformer, such as `shifted windows, relative position encoding`, etc.**

Our code is based on Swin-Unet, if you need to use it, please be sure to reference us and Swin-Unet(https://arxiv.org/abs/2105.05537 or https://mcv-workshop.github.io/)!

I hope this will help you to reproduce the results.

## 1. Download pre-trained swin transformer model (Swin-T)
following 
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
sh train.sh or python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

