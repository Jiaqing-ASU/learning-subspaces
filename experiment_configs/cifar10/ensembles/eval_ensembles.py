#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import random
import sys

sys.path.append(os.path.abspath("."))
from args import args
from main import main as run


def samples(n, k):

    A = list(range(n))
    m = 3

    samples = set()

    if n == k:
        m = 1
    tries = 0
    while len(samples) < m:
        samples.add(tuple(sorted(random.sample(A, k))))
        tries += 1

    return samples


def main():
    train_epochs = 80
    args.label_noise = 0.0

    # TODO: change these paths -- this is an example.
    args.data = "~/data"
    args.log_dir = "learning-subspaces-results/cifar/eval-ensemble"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    for num_models in [1, 2, 3]:

        to_try = samples(3, num_models)

        for seed in range(len(to_try)):

            next_try = to_try.pop()

            args.seed = seed
            args.set = "CIFAR10"
            args.multigpu = [0]
            args.model = "CIFARResNet"
            args.model_name = "cifar_resnet_20"
            args.conv_type = "StandardConv"
            args.bn_type = "StandardBN"
            args.conv_init = "kaiming_normal"
            args.trainer = "ensemble"
            args.epochs = 0
            args.resume = [
                f"learning-subspaces-results/cifar/train-ensemble-members/"
                f"id=base+ln={args.label_noise}+seed={c}"
                f"+try=0/epoch_{train_epochs}_iter_{train_epochs * round(50000 / 128)}.pt"
                for c in next_try
            ]
            args.num_models = len(args.resume)
            args.name = (
                f"id=ensmeble+ln={args.label_noise}+epochs={train_epochs}"
                f"+num_models={args.num_models}+seed={seed}"
            )

            args.save = False
            args.save_data = True
            args.pretrained = True

            args.batch_size = 128
            args.output_size = 1000
            args.trainswa = False
            args.label_smoothing = None
            args.device = "cpu"
            args.optimizer = "sgd"
            args.momentum = 0
            args.wd = 1e-4
            args.lr_policy = "cosine_lr"
            args.log_interval = 100
            args.trainer = "default"
            args.lr = 0.1
            args.test_freq = 10

            run()


if __name__ == "__main__":
    main()
