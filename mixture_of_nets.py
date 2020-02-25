from __future__ import print_function
import argparse
import os
import math
import random
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from utility import dataset, make_dir, weights_init, log_display
from alpha_beta_declaration import alpha, beta


def lossf(ML_target, loss_lower_bound, HL_output, loss_weights):
    return loss_weights[0]* F.nll_loss(loss_lower_bound, ML_target)+loss_weights[1]*F.kl_div(torch.ones_like(HL_output)/HL_output.shape[1], HL_output)


def forward_complete(model_dict, num_clusters, input_data):
    _, HL_output = model_dict['alpha'](input_data)
    HL_output_exp = torch.exp(HL_output)
    ML_output = None
    loss_lower_bound = None
    for i in range(num_clusters):

        _, ml_output = model_dict['beta_'+str(i+1)](input_data)
        coeff = HL_output_exp[:, i].unsqueeze(1).expand(-1, ml_output.size(1))
        hl_output = HL_output[:, i].unsqueeze(1).expand(-1, ml_output.size(1))

        contrib_output  = torch.mul(ml_output, coeff)
        contrib_loss    = torch.mul(torch.sub(ml_output, hl_output), coeff)

        if ML_output is None:
            ML_output = contrib_output
            loss_lower_bound = contrib_loss
        else:
            ML_output += contrib_output
            loss_lower_bound += contrib_loss
    return ML_output, HL_output_exp, ML_output


def train(model_dict, num_clusters, writer, train_loader, optim_dict, epoch, loss_weights, CUDA_DEVICE, results_address, train_alpha=False):
    for key in model_dict.keys():
        model_dict[key].train()

    accumulated_loss = 0
    accumulated_correct = 0

    predicted_tuple = tuple()
    label_tuple     = tuple()
    
    for batch_idx, sample in enumerate(tqdm(train_loader)):
        
        data = Variable(sample['data']).cuda(CUDA_DEVICE)
        #HL_target, ML_target = Variable(sample['context']).cuda(CUDA_DEVICE), Variable(sample['label']).cuda(CUDA_DEVICE)
        ML_target = Variable(sample['label']).cuda(CUDA_DEVICE)

        
        nets_to_optimize = ['beta_'+str(i+1) for i in range(num_clusters)]
        
        if train_alpha:
            nets_to_optimize =  nets_to_optimize + ['alpha']

        for key in nets_to_optimize:
            optim_dict[key].zero_grad()
            
            ML_output, HL_output, loss_lower_bound = forward_complete(model_dict, num_clusters, data) 
            loss = lossf(ML_target, loss_lower_bound, HL_output, loss_weights)
            loss.backward()
            optim_dict[key].step()

        ML_output, HL_output, loss_lower_bound = forward_complete(model_dict, num_clusters, data)
        loss = lossf(ML_target, loss_lower_bound, HL_output, loss_weights)
        accumulated_loss += loss.data.item() * len(data)

        pred = ML_output.max(1, keepdim=True)[1]
        correct = pred.eq(ML_target.view_as(pred)).sum().data.item()

        accumulated_correct += correct 

        predicted_tuple = predicted_tuple + (pred.cpu().numpy(), )
        label_tuple = label_tuple + (sample['label'].view_as(pred).numpy(), )

    accumulated_loss = accumulated_loss/ len(train_loader.dataset)
    accumulated_correct = accumulated_correct / len(train_loader.dataset)
  
    predicted = np.vstack(predicted_tuple)
    labels    = np.vstack(label_tuple)
    f1_score_ = f1_score (y_true = labels, y_pred = predicted, average = 'weighted', labels=np.unique(predicted))
    
    writer.add_scalar('train_loss', accumulated_loss, epoch)
    writer.add_scalar('train_accuracy', accumulated_correct, epoch)
    writer.add_scalar('train_f1_score', f1_score_, epoch)

    log_display (mode = 'accuracy', data_dict = {'mode':'training', 'accumulated_loss': accumulated_loss, 'accumulated_correct': accumulated_correct}, results_address = results_address)

    return accumulated_correct


def test(model_dict, num_clusters, writer, test_loader, epoch, loss_weights, CUDA_DEVICE, results_address, bootstrap_number):
    for key in model_dict.keys():
        model_dict[key].eval()

    accumulated_loss = 0
    accumulated_correct = 0

    predicted_tuple = tuple()
    label_tuple     = tuple()
 
    for batch_idx, sample in enumerate(tqdm(test_loader)):
        
        data = Variable(sample['data']).cuda(CUDA_DEVICE)
        #HL_target, ML_target = Variable(sample['context']).cuda(CUDA_DEVICE), Variable(sample['label']).cuda(CUDA_DEVICE)
        ML_target = Variable(sample['label']).cuda(CUDA_DEVICE)

 
        ML_output, HL_output, loss_lower_bound = forward_complete(model_dict, num_clusters, data)
        accumulated_loss += lossf(ML_target, loss_lower_bound, HL_output, loss_weights).data.item() * len(data)

        pred = ML_output.max(1, keepdim=True)[1]
        accumulated_correct += pred.eq(ML_target.view_as(pred)).sum().data.item()
        predicted_tuple = predicted_tuple + (pred.cpu().numpy(), )
        label_tuple = label_tuple + (sample['label'].view_as(pred).numpy(), )

    accumulated_loss /= len(test_loader.dataset)
    accumulated_correct = accumulated_correct / float(len(test_loader.dataset))
    predicted = np.vstack(predicted_tuple)
    labels    = np.vstack(label_tuple)
    f1_score_ = f1_score (y_true = labels, y_pred = predicted, average = 'weighted', labels=np.unique(predicted))
    

    #make_dir (os.path.join(results_address, '/mixture_bootstrap_%d/'%bootstrap_number)) 
    pd.DataFrame(data = np.hstack((labels, predicted)), columns = ['label','predicted']).to_csv(os.path.join(make_dir(os.path.join(results_address, 'mixture_bootstrap_%d'%bootstrap_number)), 'labels_vs_predicted_epoch_%d.csv'%epoch))
  
    writer.add_scalar('test_f1_score', f1_score_, epoch)
    writer.add_scalar('test_loss', accumulated_loss, epoch)
    writer.add_scalar('test_accuracy', accumulated_correct, epoch)

    log_display (mode = 'accuracy', data_dict = {'mode':'testing', 'accumulated_loss': accumulated_loss, 'accumulated_correct': accumulated_correct}, results_address = results_address)

    return accumulated_correct


def extract_alpha_output (model, num_clusters, dataset_loader, bootstrap_number, cache_address, CUDA_DEVICE):

    model.eval()

    features = []
    for batch_idx, sample in enumerate(dataset_loader):#(tqdm(dataset_loader)):
        data = Variable(sample['data']).cuda(CUDA_DEVICE)
        target = Variable(sample['context']).cuda(CUDA_DEVICE)

        _, output = model (data)
        features.append (np.append (output.detach ().cpu ().data, target+1))

    np.save (os.path.join(cache_address, 'alpha_output_fold_%d.npy'%(bootstrap_number)), np.asarray (features))


def alpha_output_extractor (alpha_network, num_clusters, bootstrap_number, data_address, cache_address, CUDA_DEVICE):
    train_loader = torch.utils.data.DataLoader (dataset (data_address, {'train': "info_train_fold_%d.csv", 'test':"info_test_fold_%d.csv"}, True, bootstrap_number), batch_size=1, shuffle=False)
    model = alpha_network

    extract_alpha_output (model, num_clusters, train_loader, bootstrap_number, cache_address, CUDA_DEVICE)

    del model


def create_model_optim_sched_dicts(num_clusters, bootstrap_number, optimization_params, use_pretraining, cache_address, CUDA_DEVICE):

    model_dict = {}
    optim_dict = {}
    sched_dict = {}

    alpha_net=alpha(num_clusters)

    if use_pretraining:
        state_dict=torch.load (os.path.join(cache_address, 'selector_pretrain_fold_%d.pt'%(bootstrap_number)), map_location='cpu')
        state_dict = {k: state_dict[k] for k in alpha_net.state_dict()}
        alpha_net.load_state_dict(state_dict)

    alpha_net.cuda(CUDA_DEVICE)
    
    model_dict['alpha']=alpha_net
    optim_dict['alpha']=optim.SGD(alpha_net.parameters(), lr=optimization_params['lr'], momentum=optimization_params['momentum'])
    sched_dict['alpha']=lr_scheduler.ReduceLROnPlateau(optim_dict['alpha'], mode='max', verbose=True, threshold=0.02, cooldown=5)

    for i in range(num_clusters):
        beta_net=beta()
        beta_net.cuda(CUDA_DEVICE)
        beta_optim = optim.SGD(beta_net.parameters(), lr=optimization_params['lr'], momentum=optimization_params['momentum'])
        beta_sched = lr_scheduler.ReduceLROnPlateau(beta_optim, mode='max', verbose=True, threshold=0.02, cooldown=5)

        model_dict['beta_'+str(i+1)] = beta_net
        optim_dict['beta_'+str(i+1)] = beta_optim
        sched_dict['beta_'+str(i+1)] = beta_sched

    return model_dict, optim_dict, sched_dict


def mixture_training (num_clusters, bootstrap_number, optimization_params, use_pretraining, loss_weights, data_address, cache_address, results_address, CUDA_DEVICE):
    tfboard_dir = make_dir (
        os.path.join(results_address, "mixture_bootstrap_%d"%bootstrap_number))

    writer = SummaryWriter(log_dir=tfboard_dir)

    train_loader = torch.utils.data.DataLoader(dataset(data_address, data_address, {'train': "info_train_fold_%d.csv", 'test':"info_test_fold_%d.csv"}, True, bootstrap_number), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset(data_address, data_address, {'train': "info_train_fold_%d.csv", 'test':"info_test_fold_%d.csv"}, False, bootstrap_number), batch_size=64, shuffle=False)

    model_dict, optim_dict, sched_dict = create_model_optim_sched_dicts(num_clusters, bootstrap_number, optimization_params, use_pretraining, cache_address, CUDA_DEVICE)

    beginning_time = datetime.now()
    for epoch in range(1, optimization_params['epochs']+1):

        print ("epoch %d out of %d"%(epoch, optimization_params['epochs']))

        train_accr = train(model_dict, num_clusters, writer, train_loader, optim_dict, epoch, loss_weights, CUDA_DEVICE, results_address, train_alpha=True)
        val_accr = test(model_dict, num_clusters, writer, test_loader, epoch, loss_weights, CUDA_DEVICE, results_address, bootstrap_number)
        for key in sched_dict.keys():
            sched_dict[key].step(val_accr)

        time_diff = datetime.now() - beginning_time
        end_time = (optimization_params['epochs']/epoch)*time_diff + beginning_time

        print ('TRAIN SELECTOR END TIME:', end_time)

    writer.close()

    return model_dict


def train_mixture(num_clusters, bootstrap_number, use_pretraining, results_address, data_address, cache_address, optimization_params, regularization_weight, CUDA_DEVICE):

    log_display(mode='opening', data_dict={'number of clusters':num_clusters, 'bootstrap number':bootstrap_number, 'use_pretraining':use_pretraining, 'current address': os.path.dirname(os.path.abspath(__file__)),'results_address':results_address, 'data_address':data_address, 'cache_address':cache_address, 'regularizatoin weight': regularization_weight, 'CUDA device': CUDA_DEVICE, '# training epochs':optimization_params['epochs'], 'learning rate':optimization_params['lr'], 'momentum':optimization_params['momentum'], 'step size':optimization_params['step']}, results_address = results_address)


    model_dict = mixture_training (num_clusters, bootstrap_number, optimization_params, use_pretraining, [1, regularization_weight], data_address, cache_address, results_address, CUDA_DEVICE)

    with open(os.path.join(cache_address, 'model_dict_fold_%d.npy'%(bootstrap_number)), 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #alpha_output_extractor (model_dict['alpha'], num_clusters, bootstrap_number, data_address, cache_address, CUDA_DEVICE)


'''
def main():

    num_clusters = 4
    bootstrap_number = 1
    use_pretraining = True
    results_address = './results/'
    data_address = './data/'
    cache_address = './CACHE/'

    optimization_params = {'epochs': 1, 'lr': 0.001, 'momentum': 0.9, 'step': 30}

    regularization_weight = 5/10
    CUDA_DEVICE = 7


    log_display(mode='opening', data_dict={'number of clusters':num_clusters, 'bootstrap number':bootstrap_number, 'use_pretraining':use_pretraining, 'current address': os.path.dirname(os.path.abspath(__file__)),'results_address':results_address, 'data_address':data_address, 'cache_address':cache_address, 'regularizatoin weight': regularization_weight, 'CUDA device': CUDA_DEVICE, '# training epochs':optimization_params['epochs'], 'learning rate':optimization_params['lr'], 'momentum':optimization_params['momentum'], 'step size':optimization_params['step']}, results_address = results_address)


    train_mixture (num_clusters, bootstrap_number, use_pretraining, results_address, data_address, cache_address, optimization_params, regularization_weight, CUDA_DEVICE)

main()
'''
