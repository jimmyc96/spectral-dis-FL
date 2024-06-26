# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import random
import torch

from options import args_parser
from util.update import LocalUpdate, LocalUpdate_per, globaltest, localtest
from util.fedavg import  FedAvg_noniid
# from util.util import add_noise
from util.dataset import *
from util.data import *
from torch.utils.data import Dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

if __name__ == '__main__':
    # parse args
    args = args_parser()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists("./results/"):  #for fedavg, beta = 0, 
        os.makedirs("./results/")
    rootpath = "./results/ratio_%.1f_lambda_%.1f_%.1f_LP_%.1f" % (args.ratio, args.lambda_g, args.lambda_l, args.beta)
    # if args.beta > 0: # set default mu = 1, and set beta = 1 when using fedprox
    #     #args.mu = 1
    #     rootpath += "_LP_%.1f" % (args.beta)
    f_acc = open(rootpath + '_acc_{}_{}_cons_frac{}_iid{}_iter{}_ep{}_lr{}_N{}_{}_seed{}_p{}_dirichlet{}.txt'.format(
        args.dataset, args.model, args.frac, args.iid, args.rounds, args.local_ep, args.lr, args.num_users, args.num_classes, args.seed, args.non_iid_prob_class, args.alpha_dirichlet), 'a')

    # dataset_train, dataset_test, dict_users = get_dataset(args)

    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(datasetObj.get_args())  
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(datasetObj.get_args())  
     

    # build model
    netglob = build_model(args)

    # copy weights
    w_glob = netglob.state_dict()  # return a dictionary containing a whole state of the module
    
    l_model=netglob
    p_net_list = [l_model.state_dict()]*args.num_users
    
    # training
    loss_train = []
    acc_test = []

    m = max(int(args.frac * args.num_users), 1) #num_select_clients
    prob = [1/args.num_users for j in range(args.num_users)]

    # add fl training
    for rnd in range(args.rounds):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        
        for idx in idxs_users:  # training over the subset 
            acc_per_s  = []

            l_model.load_state_dict(copy.deepcopy(p_net_list[idx]))
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed, w_g=netglob.to(args.device), epoch=args.local_ep)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))
            
            # local_p = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            localp = LocalUpdate_per(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local_p, loss_local = localp.update_weights(net=copy.deepcopy(l_model).to(args.device), seed=args.seed, w_g=netglob.to(args.device), epoch=args.local_ep)
            
            p_net_list[idx]=copy.deepcopy(w_local_p)
            l_model.load_state_dict(copy.deepcopy(w_local_p))
            acc_l =  localtest(copy.deepcopy(l_model).to(args.device), DatasetSplit(dataset_test, dict_localtest[idx]) ,args)
            acc_per_s.append(acc_l)
            acc_per = sum(acc_per_s)/len(acc_per_s)
            
        
        # w_glob = FedAvg(w_locals) # global averaging
        # if args.iid:
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob))

        acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
                
        f_acc.write("round %d, global test acc  %.4f \n"%(rnd, acc_s2))
        f_acc.write("round %d, average localtest acc  %.4f \n"%(rnd, acc_per))
        f_acc.flush()
        print('round %d, global test acc  %.3f, local test acc  %.3f \n'%(rnd, acc_s2, acc_per))


        
            

    torch.cuda.empty_cache()
