# vision-transformer-implementation
Vision Transformer implementation from scratch using Tensorflow (For education purpose)

## Introduction
This github repository implements the Vision Transformer Architecture used for image recognition 
stated in [this paper](https://arxiv.org/abs/2010.11929). The experiments are conducted on the 
dog-and-cat binary classification dataset.

## Instructions

## References
- An image is worth 16x16 words : Transformers for Image Recognition : [Paper](https://arxiv.org/abs/2010.11929)
- Dog and cat classification dataset : [Link](https://www.kaggle.com/c/dogs-vs-cats)

## TODO
- [x] Load images and resize them to 64 x 64 dimension.
- [x] Divide each image into 16 16x16 patches (Four patches on each dimension).
- [x] Flatten the patches -> Final dimension = (batch_size, 16, 16*16).
- [x] Create the positional embedding layer.
- [x] Create the layer norm layer.
- [x] Create the Single-headed self-attention module.
- [x] Create the Multi-headed attention module.
- [x] Create a complete Transformer Encoder architecture.
- [x] Write train-step function and train function.
- [ ] Run experiments and export results to wandb.ai.
