# Training

We provide FeTA2021 training, FLARE2021 training, and AMOS2022 Finetuning commands here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## FeTA 2021 & FLARE 2021 Training
Note that **'MemorySwinUNETR' denotes 'SwinUNETR-MAM' and 'MemorynnFormer' denotes 'nnFormer-MAM'**.

SwinUNETR-MAM and nnFormer-MAM training on FeTA 2021 with a single GPU:
```
python main_train.py --root root_folder_path --output output_folder_path \
--dataset feta --network MemorySwinUNETR --mode train --pretrain False \
--batch_size 1 --crop_sample 2 --lr 0.0001 --optim AdamW --max_iter 40000 \ 
--eval_step 500 --gpu 0 --cache_rate 1.0 --num_workers 2
```

SwinUNETR-MAM and nnFormer-MAM training on FLARE 2021 with a single GPU:
```
python main_train.py --root root_folder_path --output output_folder_path \
--dataset flare --network MemorynnFormer --mode train --pretrain False \
--batch_size 1 --crop_sample 2 --lr 0.0001 --optim AdamW --max_iter 40000 \ 
--eval_step 500 --gpu 0 --cache_rate 0.2 --num_workers 2
```

- If the error "Out of GPU memory" is popped out, please reduce the number of crop_sample or cache_rate 
- We perform 40000 iterations for training, and validation is performed in every 500 step.
- For the user with GPU memory <= 16Gb, we recommend to separate training and validation process (save all model weights and perform validation afterwards).
- If you want to run our code with your dataset, please look into [load_datasets_transforms.py](load_datasets_transforms.py) and you can directly create new transforms following the similar format for your own dataset. 



## Network Choices
We also aim to perform benchmarking in the latest public multi-organ/tissue segmentation datasets with volumetric transformer network SOTA. This GitHub allows to perform fair comparisons for network by using same data augmentation and preprocessing. Here is the summary of the available network in our code:
- MemorySwinUNETR
- MemorynnFormer
- 3D U-Net
- SegResNet
- TransBTS
- UNETR
- nnFormer
- SwinUNETR
- 3D UX-Net

Feel free to provide recommendations of adding latest volumetric transformer or CNN networks and we can further implement it for benchmarking.

