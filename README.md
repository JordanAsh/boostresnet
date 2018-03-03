# BoostResNet

This repository contains a simple implementation of the article [Learning Deep ResNet Blocks Sequentially using Boosting Theory](https://arxiv.org/abs/1706.04964).

The program `brn.py` assumes the existence of a dataset in torch format that is already normalized. It uses a 50-layer ResNet architecture from [Facebook](https://github.com/facebook/fb.resnet.torch) that takes 32 x 32 images as input, but can be easily modified to accomodate other architectures or data.

`python brn.py --data CIFAR.t7 --transform`
