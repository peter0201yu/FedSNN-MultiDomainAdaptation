import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from models.model import VGG5_ANN, VGG5_SNN_BNTT
from models.test import test_img
from scipy import stats
import datetime
from collections import namedtuple
from utils.options import args_parser
from utils.lib import seed_everything, DatasetSplit, normalize


if __name__=='__main__':

    seed_everything()

    args = args_parser()

    # Data
    if args.snn:
        source_modelpath = '/home/zy264/scratch60/FedMDA/MDA/pretrained_source/SNN/2022-07-22.pth'
    else:
        source_modelpath = '/home/zy264/scratch60/FedMDA/MDA/pretrained_source/ANN/2022-07-20.pth'

    target_train_datapath = '/home/zy264/scratch60/FedMDA/digit5/' + args.target + '/train_images'
    target_test_datapath = '/home/zy264/scratch60/FedMDA/digit5/' + args.target + '/test_images'

    if args.target == 'syn' or args.target == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.target == 'mnist_m' or args.target == 'usps':
        transform = transforms.Compose([
            # pad 28 x 28 imgs to 32 x 32
            transforms.ToTensor(),
            transforms.Pad(2)
        ])

    target_train_dataset = datasets.ImageFolder(target_train_datapath, transform=transform)
    print("Total data: ", len(target_train_dataset))
    # target_train_dl = DataLoader(target_train_dataset, batch_size=32, shuffle=True, drop_last=True)
    target_test_dataset = datasets.ImageFolder(target_test_datapath, transform=transform)
    target_test_dl = DataLoader(target_test_dataset, batch_size=32, shuffle=True, drop_last=True)

    # Model
    # ======= pre-trained source network =======
    if args.snn:
        net = VGG5_SNN_BNTT().cuda()
    else:
        net = VGG5_ANN().cuda()

    net.load_state_dict(torch.load(source_modelpath))
    print("Model initialized using pretrained model")

    epochs = 40
    eval_every = 1
    batch_size = 32

    threshold = 1.8
    print("Threshold: ", threshold)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001) # use default momentum, weight decay
    loss_func = torch.nn.CrossEntropyLoss()

    epoch_loss = []
    test_accuracies = []

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # First forward all data samples and sort them by self entropy
        net.eval()
        # sample_entropies = {}

        forward_dl = DataLoader(target_train_dataset, batch_size=32, shuffle=False, drop_last=True)
        selected_samples = []
        pseudo_labels = []

        for idx, (imgs, labels) in enumerate(forward_dl):
            probs = net(imgs.cuda()).cpu().detach()
            for i in range(batch_size):
                if min(np.array(probs[i])) < 0:
                    biased = np.array(probs[i]) - min(np.array(probs[i]))
                else:
                    biased = np.array(probs[i])

                if stats.entropy(biased) < threshold:
                    selected_samples.append(idx * batch_size + i)

                y_pred = np.where(biased == np.amax(biased))[0][0]
                pseudo_labels.append(y_pred)


        print("Selected %d samples" % len(selected_samples))

        confident_train_dataset = DatasetSplit(target_train_dataset, selected_samples, pseudo_labels=pseudo_labels)
        train_dl = DataLoader(confident_train_dataset, batch_size=32, shuffle=True, drop_last=True)
        
        net.train()

        batch_loss = []
        for batch_idx, (images, labels) in enumerate(train_dl):
            images, labels = images.cuda(), labels.cuda()
            net.zero_grad()
            log_probs = net(images)

            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print("Epoch loss: ", epoch_loss[-1])

        if epoch % eval_every == 0:
            print("Testing")
            net.eval()
            acc_test, loss_test = test_img(net, target_test_dataset, args)
            print("Round {:d}, Testing accuracy: {:.2f}".format(epoch, acc_test))
            test_accuracies.append(acc_test)
    
    f = open("./test_accuracies.txt", "w")
    for i in range(len(test_accuracies)):
        f.write("Round {}, Acc: {} \n".format((i+1)*eval_every, test_accuracies[i]))
