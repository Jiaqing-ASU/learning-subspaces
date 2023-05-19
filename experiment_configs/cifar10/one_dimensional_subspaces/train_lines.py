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

    for seed in range(2):
        args.seed = seed
        args.lr = 0.1
        args.label_noise = 0.0
        args.beta = 1.0
        args.layerwise = False
        args.num_samples = 1

        args.test_freq = 10
        args.set = "CIFAR10"
        args.multigpu = [0]
        args.model = "CIFARResNet"
        args.model_name = "cifar_resnet_20"
        args.conv_type = "LinesConv"
        args.bn_type = "LinesBN"
        args.conv_init = "kaiming_normal"
        args.trainer = "train_one_dim_subspaces"
        args.epochs = 80
        args.warmup_length = 5
        args.data_seed = 0
        args.train_update_bn = True
        args.update_bn = True

        args.batch_size = 128
        args.num_models = 2
        args.output_size = 1000
        args.trainswa = False
        args.resume = False
        args.label_smoothing = None
        args.device = "cpu"
        args.optimizer = "sgd"
        args.momentum = 0
        args.wd = 1e-4
        args.lr_policy = "cosine_lr"
        args.log_interval = 100

        args.name = (
            f"id=lines+ln={args.label_noise}"
            f"+beta={args.beta}"
            f"+num_samples={args.num_samples}"
            f"+seed={args.seed}"
        )

        args.save = True
        args.save_epochs = []
        args.save_iters = []

        # TODO: change these paths -- this is an example.
        args.data = "~/data"
        args.log_dir = (
            "learning-subspaces-results/cifar/one-dimesnional-subspaces"
        )

        run()
