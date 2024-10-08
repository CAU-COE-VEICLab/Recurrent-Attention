### nnFormer-RecA & SwinUNETR-RecA

**nnFormer-RecA and SwinUNETR-RecA are to replace all the Self-Attention in the above encoder with the Recurrent-Attention-H. Note that we only replace Self-Attention and keep the design of Swin-Transformer, such as `shifted windows, relative position encoding`, etc.**

Our code is based on 3D UX-Net, if you need to use it, please be sure to reference us and 3D UX-Net(https://arxiv.org/abs/2209.15076)!

I hope this will help you to reproduce the results.

 ## Installation
 Please look into the [INSTALL.md](INSTALL.md) for creating conda environment and package installation procedures.

 ## Training Tutorial
 - [x] FeTA 2021, FLARE 2021 Training Code [TRAINING.md](TRAINING.md)
 
 (Feel free to post suggestions in issues of recommending latest proposed transformer network for comparison. Currently, the network folder is to put the current SOTA transformer. We can further add the recommended network in it for training.)
 

<!-- ✅ ⬜️  -->
## Training
Training and fine-tuning instructions are in [TRAINING.md](TRAINING.md). Pretrained model weights will be uploaded for public usage later on.

<!-- ✅ ⬜️  -->
## Evaluation
Note that **'SwinUNETRRecA' denotes 'SwinUNETR-RecA' and 'nnFormerRecA' denotes 'nnFormer-RecA'**.

Using SwinUNETR-RecA as an example, Efficient evaluation can be performed for the above two public datasets as follows:
```
python test_seg.py --root path_to_image_folder --output path_to_output \
--dataset flare --network SwinUNETRRecA --trained_weights path_to_trained_weights \
--mode test --sw_batch_size 4 --overlap 0.7 --gpu 0 --cache_rate 0.2 \
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## References
* [3D UX-Net](https://github.com/MASILab/3DUX-Net)

 
 


