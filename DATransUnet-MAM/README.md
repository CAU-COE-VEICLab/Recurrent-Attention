# DA-TransUnet-MAM
**DA-TransUnet-MAM is to replace all the Self-Attention in DA-TransUnet's bottleneck with the Memory-Attention and apply the Stage Transmission Strategy (STS). Note that we only replace Self-Attention and keep the design of Vision-Transformer, such as `position encoding`, etc.**

Our code is based on DA-TransUnet, if you need to use it, please be sure to reference us and DA-TransUnet(https://arxiv.org/abs/2310.12570)!

I hope this will help you to reproduce the results.

### 1.Prepare pre-trained ViT models
following [DA-TransUnet](https://github.com/SUN-1024/DA-TransUnet): R50-ViT-B_16, At the same time, the parameter file (.pth) in the paper is also stored.
* (You can download and compress it, put it into the model file and rename it MA_Synapse224, and then use the test code (python test.py --dataset Synapse --vit_name R50-ViT-B_16) to get the test results.)

### 2.Prepare data
The preprocessed data is following DA-TransUnet. 
[please click this link to download](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing)

### 3.Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4.Train/Test
Run the train script on synapse dataset. The batch size can be reduced to 12 or 16 to save memory(please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference 
* [DA-TransUnet](https://github.com/SUN-1024/DA-TransUnet)
