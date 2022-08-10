import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from scipy import stats
import copy
import datetime

from utils.options import args_parser
from utils.lib import seed_everything, DatasetSplit, normalize
from utils.sampling import iid, non_iid

from models.model import VGG5_ANN, VGG5_SNN_BNTT
from models.test import test_img
from models.Fed import FedLearn
from models.Update import LocalUpdate

import wandb

if __name__ == '__main__':

    seed_everything()

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if args.wandb:
        wandb.init(project="FedSNN-MDA", name=args.wandb,
                    config={"snn": True if args.snn else False, "iid": True if args.iid else False, \
                            "global epochs": args.epochs, "local epochs": args.local_ep, \
                            "num_users_per_domain": args.num_users_per_domain, \
                            "mda_threshold": args.mda_threshold, "mda_threshold_frac": args.mda_threshold_frac,\
                            "timesteps": args.timesteps if args.snn else None, "alpha": None if args.iid else args.alpha})

    # load datasets and split users
    train_datasets, test_datasets = {}, {}
    if args.dataset == "digit5":
        # digit5 domains: mnist_m, syn, svhn, usps
        domain_names = ["mnist_m", "syn", "svhn", "usps"]
        train_datapaths = {domain_name : "/home/zy264/scratch60/FedMDA/digit5/{}/train_images".format(domain_name) for domain_name in domain_names}
        test_datapaths = {domain_name : "/home/zy264/scratch60/FedMDA/digit5/{}/test_images".format(domain_name) for domain_name in domain_names}
        
        for domain_name in ["syn", "svhn"]:
            transform = transforms.Compose([transforms.ToTensor()])
            train_datasets[domain_name] = datasets.ImageFolder(train_datapaths[domain_name], transform=transform)
            test_datasets[domain_name] = datasets.ImageFolder(test_datapaths[domain_name], transform=transform)
        for domain_name in ["mnist_m", "usps"]:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
            train_datasets[domain_name] = datasets.ImageFolder(train_datapaths[domain_name], transform=transform)
            test_datasets[domain_name] = datasets.ImageFolder(test_datapaths[domain_name], transform=transform)
    
    elif args.dataset == "office31":
        # office 31 domains: amazon, webcam, dslr
        domain_names = ["webcam", "dslr"]
        datapaths = {domain_name : "/home/zy264/scratch60/FedMDA/office31/{}".format(domain_name) for domain_name in domain_names}
        
        

    dict_users_by_domain = {}
    for domain_name in domain_names:
        if args.iid:
            dict_users_by_domain[domain_name] = iid(train_datasets[domain_name], args.num_users_per_domain)
        else:
            dict_users_by_domain[domain_name] = non_iid(train_datasets[domain_name], 10, args.num_users_per_domain)
    

    # model
    if args.snn:
        net_glob = VGG5_SNN_BNTT(timesteps=args.timesteps).cuda()
        source_modelpath = '/home/zy264/scratch60/FedMDA/MDA/pretrained_source/SNN/2022-07-22.pth'
    else:
        net_glob = VGG5_ANN().cuda()
        source_modelpath = '/home/zy264/scratch60/FedMDA/MDA/pretrained_source/ANN/2022-07-20.pth'

    net_glob.load_state_dict(torch.load(source_modelpath))
    print ("Global model initialized using pretrained model")
    # net_glob = nn.DataParallel(net_glob)

    # training
    # ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []

    # define Fed Learn object
    fl = FedLearn(args)

    for epoch in range(args.epochs):
        print("Round {} --------------------------------".format(epoch+1))

        net_glob.train()
        w_locals_all, loss_locals_all = [], []

        for domain_name in domain_names:
            for idx in range(args.num_users_per_domain):
                print("Domain: {}, user idx: {}".format(domain_name, idx))

                local = LocalUpdate(args=args, dataset=train_datasets[domain_name], idxs=dict_users_by_domain[domain_name][idx])
            
                model_copy = type(net_glob)()
                model_copy.load_state_dict(net_glob.state_dict())
                w, loss = local.mda_train(net=model_copy.to(args.device))
                w_locals_all.append(copy.deepcopy(w))
                loss_locals_all.append(copy.deepcopy(loss))

        train_loss_avg = sum(loss_locals_all) / len(loss_locals_all)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch+1, train_loss_avg))
        if args.wandb:
            wandb.log({"Average_train_loss": train_loss_avg, "Round": epoch+1})

        w_glob = fl.FedAvg(w_locals_all, w_init=net_glob.state_dict())
        net_glob.load_state_dict(w_glob)

        if epoch % args.eval_every == 0:
            # testing
            net_glob.eval()
            test_accs, test_losses = [], []
            for domain_name in domain_names:
                acc, loss = test_img(net_glob, test_datasets[domain_name], args)
                print("Round {:d}, Domain: {}, Testing accuracy: {:.2f}".format(epoch+1, domain_name, acc))
                test_accs.append(acc)
                test_losses.append(loss)

                if args.wandb:
                    wandb.log({"server_{}_test_loss".format(domain_name): loss, "server_{}_test_acc".format(domain_name): acc, "Round": epoch+1})

            # Add metrics to store
            avg_loss = sum(test_losses)/len(test_losses)
            avg_acc = sum(test_accs)/len(test_accs)
            ms_acc_test_list.append(avg_acc)
            ms_loss_test_list.append(avg_loss)
            if args.wandb:
                wandb.log({"Server_avg_test_loss": avg_loss, "Server_avg_test_acc": avg_acc, "Round": epoch+1})
    
    f = open("./{}_{}_test_acc.txt".format("snn" if args.snn else "ann", str(args.mda_threshold)), "w")
    for i in range(len(ms_acc_test_list)):
        f.write("Round {}, Loss: {:.2f}, Acc: {:.2f} \n".format((i+1)*args.eval_every, ms_loss_test_list[i], ms_acc_test_list[i]))
    f.close()
