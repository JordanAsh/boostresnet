# BoostResNet

This repository contains a simple implementation of the article [Learning Deep ResNet Blocks Sequentially using Boosting Theory](https://arxiv.org/abs/1706.04964).

This program assumes the existence of a dataset in torch format that is already normalized, and uses the a 50-layer ResNet architecture from [Facebook](https://github.com/facebook/fb.resnet.torch) that takes 32 x 32 images as input. It can be easily modified to accomodate other architectures or data. It can be run like:

`python brn.py --data CIFAR.t7 --transform`
