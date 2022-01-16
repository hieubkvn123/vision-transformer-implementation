from train_utils import train
from wandb_conf import config
from nn import TransformerEncoder

from argparse import ArgumentParser

def main(opt):
    model = TransformerEncoder(L=opt['L'], num_attn_heads=opt['num_attn_heads'],
                               d_model=opt['d_model'], batch_size=opt['batch_size'])

    wandb_log = None
    if(not opt['no_wandb']) : wandb_log = config

    train(model, opt['data_dir'], batch_size=opt['batch_size'],
          d_model=opt['d_model'], img_size=opt['img_size'], epochs=opt['epochs'],
          lr=opt['lr'], wandb_log=wandb_log)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', required=False, default='../data/DOG_CAT_SMALL/train', help='Path to data folder with sub-folders for each class')
    parser.add_argument('--L', required=False, default=8, help='Number of transformer encoder modules')
    parser.add_argument('--num_attn_heads', required=False, default=8, help='Number of attention heads')
    parser.add_argument('--d_model', required=False, default=128, help='Number of feature dimensions')
    parser.add_argument('--img_size', required=False, default=64, help='Default image size')
    parser.add_argument('--batch_size', required=False, default=16, help='Training and validating batch size')
    parser.add_argument('--epochs', required=False, default=100, help='Number of training iterations')
    parser.add_argument('--lr', required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true', help='Whether to log training info online to wandb or offline')

    opt = vars(parser.parse_args())
    

    main(opt)
