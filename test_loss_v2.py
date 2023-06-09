###
# Author: John Kevin Cava
# Date: June 7, 2023
# NOTE: Using the different alphas for the model and getting the accuracies/losses
###

import torch
import torch.nn as nn
import torch.nn.functional as F

alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for alpha in alphas:
    # Set variables
    alpha0 = alpha
    alpha1 = 1 - alpha
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
    # print(modelA.module.fc.get_weight().size())

    # Dataloader
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_dataset = torchvision.datasets.CIFAR10(
        root='~/data/cifar10/',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation on Test Data
    criterion = nn.CrossEntropyLoss()
    for data, target in val_loader:
        data, target = data.to(args.device), target.to(args.device)

        output, feats = modelA(data)
        test_loss += criterion(output, target).item()

        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader)
    test_error = 1.0 - test_acc
    print('Test Loss:',str(test_loss))
    print('Test Acc:',str(test_acc))
    print('Test Error:',str(test_error))
    print('=======')
print('Done')