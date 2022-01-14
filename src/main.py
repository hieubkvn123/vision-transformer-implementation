from train_utils import train
from nn import TransformerEncoder

data_dir = "../data/DOG_CAT_SMALL/train"


model = TransformerEncoder(L=8)
train(model, data_dir)
