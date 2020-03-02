import torch
from fastai.vision import *
from fastai.metrics import accuracy
import os
import dill

defaults.device = torch.device('cuda')

path = ''
path_model = ''

# Data augmentation: create a list of flip, rotate, zoom, warp, lighting transforms...
tfms = get_transforms()

# Create databunch from imagenet style dataset in path with images resized 224x224 and batch size equal to 64
# and validation set about 30% of the dataset 
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=224, bs=64, valid_pct=0.3).normalize(imagenet_stats)

# Get a pretrained model (resnet34) with a custom head that is suitable for our data.
learn = cnn_learner(data, models.resnet34, metrics=[accuracy])

# Fit a model following the 1cycle policy with 15 epochs
learn.fit_one_cycle(15)

# Save the model (pytorch form .pt)
torch.save(learn.model, path_model+'my_model.pt', pickle_module=dill)
#learn.export(path_model+'export.pkl')