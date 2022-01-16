# vision-transformer-implementation
Vision Transformer implementation from scratch using Tensorflow (For education purpose)

## Introduction
This github repository implements the Vision Transformer Architecture used for image recognition 
stated in [this paper](https://arxiv.org/abs/2010.11929). The experiments are conducted on the 
dog-and-cat binary classification dataset.

## Instructions
To run the script with default options, just navigate to src/ and run
```bash
python3 main.py
```

To change the run options, modify one of the following options :
```python
parser.add_argument('--data_dir', required=False, default='../data/DOG_CAT_SMALL/train', help='Path to data folder with sub-folders for each class')
parser.add_argument('--L', required=False, default=8, help='Number of transformer encoder modules')
parser.add_argument('--num_attn_heads', required=False, default=8, help='Number of attention heads')
parser.add_argument('--d_model', required=False, default=128, help='Number of feature dimensions')
parser.add_argument('--img_size', required=False, default=64, help='Default image size')
parser.add_argument('--batch_size', required=False, default=16, help='Training and validating batch size')
parser.add_argument('--epochs', required=False, default=100, help='Number of training iterations')
parser.add_argument('--lr', required=False, default=0.0001, help='Learning rate')
parser.add_argument('--no_wandb', action='store_true', help='Whether to log training info online to wandb or offline')
```

By default the training logs will be exported to wandb. To change your own wandb project, modify the src/wandb_conf.py file
```python
config = {
    'project' : 'your_wandb_project',
    'entity' : 'your_wandb_username'
}
```
Login to wandb via the CLI and run the script as usual. Otherwise, you can log your training info offline by specifying 
the --no_wandb option.

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
- [x] Run experiments and export results to wandb.ai.
