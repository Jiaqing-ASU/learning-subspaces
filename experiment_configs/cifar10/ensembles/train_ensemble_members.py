#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import sys

sys.path.append(os.path.abspath("."))

from args import args
from main import main as run

if __name__ == "__main__":

    for seed in range(3):
        args.seed = seed
        args.label_noise = 0.0

        args.set = "CIFAR10"
        # this says to use GPU 0.
        args.multigpu = [0]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "StandardConv"
        args.bn_type = "StandardBN"
        args.conv_init = "kaiming_normal"
        args.epochs = 10
        args.warmup_length = 5
        args.data_seed = 0
        args.name = f"id=base+ln={args.label_noise}+seed={args.seed}"
        args.save = True
        args.save_epochs = []
        args.save_iters = []

        args.batch_size = 128
        args.num_models = 2
        args.output_size = 1000
        args.trainswa = False
        args.resume = False
        args.label_smoothing = None
        args.device = "cuda"
        args.optimizer = "sgd"
        args.momentum = 0
        args.wd = 1e-4
        args.lr_policy = "cosine_lr"
        args.log_interval = 100
        args.trainer = "default"
        args.lr = 0.1
        args.test_freq = 10
        args.workers = 16
        
        # TODO: change these paths -- this is an example.
        args.data = "~/data"
        args.log_dir = "learning-subspaces-results/cifar/train-ensemble-members"

        run()
