import torch
import sys
import pickle
import foolbox
from torch.utils.serialization import load_lua
import pdb
import os
import numpy as np
import torchfile
import scipy.io
from torch import nn
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gammaFirst', default=0.5,  help='initial gamma')
parser.add_argument('--checkEvery', default=10000, help='how frequently to check gamma requirement - 10k for cifar, 5k for svhn')
parser.add_argument('--data', default='SVHN.t7', help='load data')
parser.add_argument('--gammaThresh',default=-0.0001, help='gamma threshold to stop training layer')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--maxIters', default=10000, help='maximum iterations before stopping train layer')
parser.add_argument("--transform", help='do CIFAR-style image transformations?', action="store_true")
parser.add_argument('--printEvery', default=100,  help='how frequently to print')
parser.add_argument('--batchSize', default=100,  help='batch size')
parser.add_argument('--modelPath', default='model.pt',  help='output model')
opt = parser.parse_args()

data = torchfile.load(opt.data) # load dataset in torch fromat (assuming already normalized)
Xtrain = data.Xtrain
Ytrain = data.Ytrain - 1
Xtest = data.Xtest
Ytest = data.Ytest - 1
cut = int(np.shape(Xtrain)[0] / opt.batchSize * opt.batchSize) # cut off a few samples for simplicity
nTrain = cut
Xtrain = Xtrain[:cut]
Ytrain = Ytrain[:cut]
cut = int(np.shape(Xtest)[0] / opt.batchSize * opt.batchSize)
Xtest = Xtest[:cut]
Ytest = Ytest[:cut]
nTest = cut
numClasses = 10

# print helper (python3 print doesn't play work well with slurm)
def printer(print_arr):
    for v in print_arr: sys.stdout.write(str(v) + '\t')
    sys.stdout.write('\n')
    sys.stdout.flush()

# load fb resnet
import fbrn
model = fbrn.tmp
model.load_state_dict(torch.load('fbrn.pth'))

# build blocks (specific to fb architecture)
allBlocks = {}
allBlocks[0] = nn.Sequential(model[0], model[1], model[2])
for i in range(8): allBlocks[1 + i] = model[3][i] 
for i in range(8): allBlocks[9 + i] = model[4][i] 
for i in range(8): allBlocks[17+ i] = model[5][i]
criterion = nn.CrossEntropyLoss().cuda()
nFilters = 15; rounds = 25

# helper for augmentation - necessary for cifar 
def transform(X):
    tmp = np.zeros((np.shape(X)[0],3,38,38))
    tmp[:, :, 2:34, 2:34] = X
    for i in range(np.shape(X)[0]):
        r1 = np.random.randint(4)
        r2 = np.random.randint(4)
        X[i] = tmp[i, :, r1 : r1 + 32, r2 : r2 + 32]
        if np.random.uniform() > .5:
            X[i] = X[i,:,:,::-1]
    return X

# helper for model evaluation
def getPerformance(net, X, Y, n):
    acc = 0.
    model.eval()
    Xoutput = np.zeros((X.shape[0], 10))
    for batch in range(int(X.shape[0] / opt.batchSize)):
        start = batch * opt.batchSize; stop = (batch + 1) * opt.batchSize - 1
        ints = np.linspace(start, stop, opt.batchSize).astype(int)
        data = Variable(torch.from_numpy(X[ints])).float().cuda()
        for i in range(n): data = allBlocks[i](data)
        output = net(data)
        acc += np.mean(torch.max(output,1)[1].cpu().data.numpy() == Y[ints])
        Xoutput[ints] = output.cpu().data.numpy()
    acc /= (X.shape[0] / opt.batchSize)
    model.train()
    return acc, Xoutput

# initialize model statistics
a_previous = 0.0
a_current = -1.0
s = np.zeros((nTrain, numClasses))
cost = np.zeros((nTrain, numClasses))
Xoutput_previous = np.zeros((nTrain, numClasses))
Ybatch = np.zeros((opt.batchSize))
YbatchTest = np.zeros((opt.batchSize))
gamma_previous = opt.gammaFirst
totalIterations = 0; tries = 0
for n in range(rounds):
    gamma = -1
    Z = 0

    # create cost function  
    for i in range(nTrain):
        localSum = 0
        for l in range(numClasses):
            if l != Ytrain[i]:
                cost[i][l] = np.exp(s[i][l] - s[i][int(Ytrain[i])])
                localSum += cost[i][l]
        cost[i][int(Ytrain[i])] = -1 * localSum
        Z += localSum

	# fetch the correct classification layers
    bk = allBlocks[n]
    ci = nn.Sequential(model[6], model[7], model[8])
    if n < 17: ci = nn.Sequential(allBlocks[17], ci)
    if n < 9:  ci = nn.Sequential(allBlocks[9], ci)
    modelTmp = nn.Sequential(bk, ci, nn.Softmax(dim=0))
    modelTmp = modelTmp.cuda()
    optimizer = torch.optim.Adam(modelTmp.parameters(), lr=opt.lr) 
    tries = 0
    XbatchTest = torch.zeros(opt.batchSize, nFilters, 32, 32)
    while (gamma < opt.gammaThresh and ((opt.checkEvery * tries) < opt.maxIters)):
        accTrain = 0; 
        accTest = 0; 
        err = 0;
        for batch in range(1, opt.checkEvery+1):
            optimizer.zero_grad()

			# get batch of training samples
            ints = np.random.random_integers(np.shape(Xtrain)[0] - 1, size=(opt.batchSize))
            Xbatch = Xtrain[ints]
            Ybatch = Variable(torch.from_numpy(Ytrain[ints])).cuda().long()

            # do transformations
            if opt.transform: Xbatch = transform(Xbatch)
            data = Variable(torch.from_numpy(Xbatch)).float().cuda()
            for i in range(n): data = allBlocks[i](data)

			# get gradients
            output = modelTmp(data)
            loss = torch.exp(criterion(output, Ybatch))
            loss.backward()
            err += loss.data[0]

			# evaluate training accuracy
            output = modelTmp(data)
            accTrain += np.mean(torch.max(output,1)[1].cpu().data.numpy() == Ytrain[ints])
			
            # get test accuracy 
            model.eval()
            ints = np.random.random_integers(np.shape(Xtest)[0] - 1, size=(opt.batchSize))
            Xbatch = Xtest[ints]
            data = Variable(torch.from_numpy(Xbatch)).float().cuda()
            for i in range(n): data = allBlocks[i](data)
            output = modelTmp(data)
            accTest += np.mean(torch.max(output,1)[1].cpu().data.numpy() == Ytest[ints])
            model.train()

            if batch % opt.printEvery == 0:
                accTrain /= opt.printEvery
                accTest  /= opt.printEvery
                err /= opt.printEvery
                printer([n, rounds, totalIterations + batch + (opt.checkEvery * tries), err, accTrain, accTest])
                accTrain = 0; accTest = 0; err = 0;

            for p in modelTmp.parameters(): p.grad.data.clamp_(-.1, .1)			
            optimizer.step()

        # compute gamma
        accTrain, Xoutput = getPerformance(modelTmp, Xtrain, Ytrain, n)
        gamma_current = -1 * np.sum(Xoutput * cost) / Z
        gamma = (gamma_current ** 2 - gamma_previous ** 2)/(1 - gamma_previous ** 2) 
        if gamma > 0: 
            gamma = np.sqrt(gamma)
        else: 
            gamma = -1 * np.sqrt(-1 * gamma)
        a_current = 0.5 * np.log((1 + gamma_current) / (1 - gamma_current))
    
        tries += 1
        if (gamma > opt.gammaThresh or ((opt.checkEvery * tries) >= opt.maxIters)):
            totalIterations = totalIterations + (tries * opt.checkEvery)
            printer([gamma, gamma_current, gamma_previous])
            printer(['a_{t+1}:', a_current, 'gamma_t:', gamma])	


    s += Xoutput * a_current - Xoutput_previous * a_previous
    accTest, _ = getPerformance(modelTmp, Xtest, Ytest, n)	
    printer(['t', rounds, 'numBatches:', tries * opt.checkEvery, 'test accuracy:', accTest])	
    gamma_previous = gamma_current

torch.save(model.state_dict(), opt.modelPath)

