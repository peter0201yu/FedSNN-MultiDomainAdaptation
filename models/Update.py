#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import sys
import os
from scipy import stats


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = DatasetSplit(dataset, idxs)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)

    def train(self, net):
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay, amsgrad = True)
        else:
            print("Invalid optimizer")

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                if self.args.verbose and (batch_idx + 1) % self.args.train_acc_batches == 0:
                    thresholds = []
                    for value in net.module.threshold.values():
                        thresholds = thresholds + [round(value.item(), 2)]
                    print('Epoch: {}, batch {}, threshold {}, leak {}, timesteps {}'.format(iter, batch_idx + 1, thresholds, net.module.leak.item(), net.module.timesteps))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def mda_train(self, net):

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr) # use default momentum, weight decay
        loss_func = torch.nn.CrossEntropyLoss()
        mda_threshold = self.args.mda_threshold
        mda_threshold_frac = self.args.mda_threshold_frac
        batch_size = self.args.bs


        epoch_loss = []
        for epoch in range(self.args.local_ep):

            # First forward all data samples and sort them by self entropy
            net.eval()

            if mda_threshold_frac == 1:
                selected_samples = [i for i in range(len(self.dataset) // batch_size * batch_size)]
            else:
                forward_dl = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, drop_last=True)
                output_entropies = []
                selected_samples = []
                for idx, (imgs, labels) in enumerate(forward_dl):
                    probs = net(imgs.cuda()).cpu().detach()
                    for i in range(batch_size):
                        # prob_abs = np.absolute(np.array(probs[i]))
                        # make sure outputs are all above zero using bias
                        if min(np.array(probs[i])) < 0:
                            biased = np.array(probs[i]) - min(np.array(probs[i]))
                        else:
                            biased = np.array(probs[i])
                        
                        etp = stats.entropy(biased)
                        if mda_threshold is not None and etp < mda_threshold:
                            selected_samples.append(i + batch_size * idx)
                        elif mda_threshold_frac is not None:
                            output_entropies.append(etp)

                if mda_threshold_frac is not None:
                    selected_number = int(len(self.dataset) * mda_threshold_frac)
                    selected_samples = sorted(range(len(output_entropies)), key=lambda i: output_entropies[i])[:selected_number]
            
            print("Selected {} samples from {} imgs".format(len(selected_samples), len(self.dataset) // batch_size * batch_size))
            
            confident_train_dataset = DatasetSplit(self.dataset, selected_samples)
            train_dl = DataLoader(confident_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
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

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
