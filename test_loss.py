###
# Author: John Kevin Cava
# Date: June 7, 2023
# NOTE: Initial script to get the data from the saved dictionary weights
###

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set variables
alpha0 = 0.5
alpha1 = 0.5
batch_size = 1

# Set device
device = 'cpu' if torch.cuda.is_available() is False else 'cuda'

# Load in the pytorch saved dictionary
PATH = '/home/jcava/learning-subspaces/learning-subspaces-results/cifar/one-dimesnional-subspaces/id=lines+ln=0.0+beta=1.0+num_samples=1+seed=1+try=0/epoch_240_iter_93840.pt'
pt_dict = torch.load(PATH,map_location=torch.device(device))

# Get models from state dict
models = pt_dict['state_dicts']
model_0 = models[0]
model_1 = models[1]

# print(model_0)
# exit()
# Get the architecture used for the ensembles
arch = pt_dict['arch']
print(arch)

from models.cifar_resnet import CIFARResNet
from args import args

args.model = "CIFARResNet"
args.model_name = "cifar_resnet_20"
args.conv_type = "LinesConv"
args.bn_type = "LinesBN"
args.conv_init = "kaiming_normal"
modelA = CIFARResNet()
modelB = CIFARResNet()

# Make the model parallel
# NOTE: Needed due to original model trained in DataParallel
# Reference: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
modelA = torch.nn.DataParallel(modelA)
modelB = torch.nn.DataParallel(modelB)

modelA.load_state_dict(model_0)
modelB.load_state_dict(model_1)

# As seen in the evaluation script 
modelA.eval()
modelA.zero_grad()
modelB.eval()
modelB.zero_grad()

modelA.apply(lambda m: setattr(m, "return_feats", True))
modelB.apply(lambda m: setattr(m, "return_feats", True))

modelA.apply(lambda m: setattr(m, "alpha", alpha1))
modelB.apply(lambda m: setattr(m, "alpha", alpha0))

print(modelA.modules)

print('Done')