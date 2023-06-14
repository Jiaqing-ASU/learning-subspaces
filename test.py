###
# Author: John Kevin Cava
# Date: June 7, 2023
# NOTE: Using the different alphas for the model and getting the accuracies/losses
###
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time

device = 'cpu' if torch.cuda.is_available() is False else 'cuda'

# Set variables
# alpha0 = alpha
# alpha1 = 1 - alpha
batch_size = 128
num_workers = 4

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

modelA.apply(lambda m: setattr(m, "alpha", 0.0))
modelB.apply(lambda m: setattr(m, "alpha", 1.0))

w2 = modelA.module.fc.get_weight().flatten()
w3 = modelB.module.fc.get_weight().flatten()

u = w2
dx = torch.norm(u)
u = u / dx

v = w3
dy = torch.norm(v)
v -= torch.dot(u, v) * u / (dy ** 2)

##
# Get (x,y) function
##
def get_xy(point, vec_x, vec_y):
    point = point.detach().numpy()
    vec_x = vec_x.detach().numpy()
    vec_y = vec_y.detach().numpy()
    return np.array([np.dot(point, vec_x), np.dot(point, vec_y)])

G = 20
alphas = np.linspace(-1.0, 1.0, G)
betas = np.linspace(-1.0, 1.0, G)

print(torch.equal(modelA.module.fc.weight,modelA.module.fc.weight))
import copy
c = copy.copy(modelA.module.fc.weight)

Z = np.zeros((G,G))
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        
        start = time.time()

        p = modelA.module.fc.get_weight()
        p = alpha * dx *  u + beta * dy * v
        modelA.module.fc.weight.data = p.view(10,64,1,1)

        # Dataloader
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_dataset = torchvision.datasets.CIFAR10(
            root='~/data/cifar10/',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Evaluation on Test Data
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output, feats = modelA(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_loader)
        Z[i,j] = test_loss

        end = time.time()
        print(i,j,'Time Taken: ', str(end-start) + 's')

plt.figure()
plt.contourf(Z)
plt.axis('scaled')
plt.colorbar()
plt.show('test.png')
