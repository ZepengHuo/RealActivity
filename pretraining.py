from __future__ import print_function
import argparse
import os
import math
import sklearn
import sklearn.cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import numpy as np
import pandas as pd
import datetime
#import torch.nn as nn
#from torch.nn import init
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

from utility import dataset, make_dir, log_display, savecsv
from alpha_beta_declaration import alpha, beta

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def lossf(target, output): 
    return F.nll_loss (output, target)


def train(model, writer, train_loader, optimizer, epoch, CUDA_DEVICE, results_address):
    model.train()

    accumulated_loss = 0
    accumulated_correct = 0

    predicted_tuple = tuple()
    label_tuple     = tuple()

    for batch_idx, sample in enumerate(tqdm(train_loader)):
        data = Variable(sample['data']).cuda(CUDA_DEVICE)
        target = Variable(sample['label']).cuda(CUDA_DEVICE)

        optimizer.zero_grad()
        _, output = model(data)
        
        loss = lossf(target, output)
        accumulated_loss += loss.data.item() * len (data)

        pred = output.max (1, keepdim=True) [1]
        correct = pred.eq (target.view_as (pred)).sum ().data.item()
        accumulated_correct += correct

        loss.backward ()
        optimizer.step ()

        predicted_tuple = predicted_tuple + (pred.cpu().numpy(), )
        label_tuple = label_tuple + (sample['label'].view_as(pred).numpy(), )


    accumulated_loss /= len(train_loader.dataset)
    accumulated_correct = accumulated_correct / float(len(train_loader.dataset))
    predicted = np.vstack(predicted_tuple)
    labels    = np.vstack(label_tuple)
    f1_score_ = f1_score (y_true = labels, y_pred = predicted, average = 'weighted', labels=np.unique(predicted))


    writer.add_scalar('train_loss', accumulated_loss, epoch)
    writer.add_scalar('train_accuracy', accumulated_correct, epoch)
    writer.add_scalar('train_f1_score', f1_score_, epoch)

    log_display (mode = 'accuracy', data_dict = {'mode':'training', 'accumulated_loss': accumulated_loss, 'accumulated_correct': accumulated_correct}, results_address = results_address)

    return accumulated_correct


def test(model, writer, test_loader, epoch, CUDA_DEVICE, results_address, bootstrap_number):
    model.eval()

    accumulated_loss = 0
    accumulated_correct = 0

    predicted_tuple = tuple()
    label_tuple     = tuple()

    for batch_idx, sample in enumerate(tqdm(test_loader)):
        data = Variable(sample['data']).cuda(CUDA_DEVICE)
        target = Variable(sample['label']).cuda(CUDA_DEVICE)
        
        _, output = model (data)
        accumulated_loss += lossf (target, output).data.item() * len(data)

        pred = output.max (1, keepdim=True) [1]
        accumulated_correct += pred.eq (target.view_as (pred)).sum ().data.item()
        predicted_tuple = predicted_tuple + (pred.cpu().numpy(), )
        label_tuple = label_tuple + (sample['label'].view_as(pred).numpy(), )

    accumulated_loss /= len(test_loader.dataset)      
    accumulated_correct = accumulated_correct / float(len(test_loader.dataset))


    predicted = np.vstack(predicted_tuple)
    labels    = np.vstack(label_tuple)
    f1_score_ = f1_score (y_true = labels, y_pred = predicted, average = 'weighted', labels=np.unique(predicted))
  
    writer.add_scalar('test_loss', accumulated_loss, epoch)
    writer.add_scalar('test_accuracy', accumulated_correct, epoch)
    writer.add_scalar('test_f1_score', f1_score_, epoch)

    pd.DataFrame(data = np.hstack((labels, predicted)), columns = ['label','predicted']).to_csv(os.path.join(make_dir(os.path.join(results_address, 'single_network_bootstrap_%d'%bootstrap_number)), 'labels_vs_predicted_epoch_%d.csv'%epoch))


    log_display (mode = 'accuracy', data_dict = {'mode':'testing', 'accumulated_loss': accumulated_loss, 'accumulated_correct': accumulated_correct}, results_address = results_address)

    return accumulated_correct


def extract_features (model, test_loader, bootstrap_number, cache_address, CUDA_DEVICE):

    model.eval()

    features = []
    for batch_idx, sample in enumerate(tqdm(test_loader)):
        data = Variable(sample['data']).cuda(CUDA_DEVICE)
        target = Variable(sample['label']).cuda(CUDA_DEVICE)

        output, _ = model (data)
        features.append (np.append (output.detach ().cpu ().data, target+1))

    np.save (os.path.join(cache_address, 'features_fold_%d.npy'%(bootstrap_number)), np.asarray (features))

    
def cluster(num_clusters, bootstrap_number, data_address, cache_address):
    content = np.load (os.path.join(cache_address, 'features_fold_%d.npy'%(bootstrap_number)))
    
    features = content [:,:-1]
    
    kmeans = sklearn.cluster.KMeans (n_clusters=num_clusters).fit(features)
  
    #print ("tooye cluster ham jaygozin")
    #filenames = pd.read_csv (os.path.join(data_address, 'info_train_fold_%d_3_removed.csv'%bootstrap_number), usecols=[0], sep=' ').sample(frac = 0.4, random_state = 1)
    #print ("cluster kamteresh kon")

    filenames = pd.read_csv (os.path.join(data_address, 'info_train_fold_%d.csv'%bootstrap_number), usecols=[0], sep=' ')

    cluster_info = []
    for i, row in filenames.reset_index(drop=True).itertuples ():
        cluster_info.append ([row, kmeans.labels_ [i]])

    info_cluster_train, info_cluster_test = train_test_split(cluster_info, test_size=0.2)
    savecsv(info_cluster_train, 'nonom', ['filename', 'label'],os.path.join(cache_address, "info_cluster_fold_%d_train.csv"%(bootstrap_number)))
    savecsv(info_cluster_test, 'nonom', ['filename', 'label'],os.path.join(cache_address, "info_cluster_fold_%d_test.csv"%(bootstrap_number)))
    
'''
def cluster(num_clusters, random_seed, bootstrap_number, data_address, cache_address):
    content = np.load (os.path.join(cache_address, 'features_fold_%d.npy'%(bootstrap_number)))

    features = content [:,:-1]
    label = content [:,-1]
    label = np.asarray ([int (l) for l in label])
    
    kmeans = sklearn.cluster.KMeans (n_clusters=num_clusters).fit (features)
    
    clabel = kmeans.labels_
    label_in_cluster = [[] for i in range (num_clusters)]
    for i in range (len (features)):
        label_in_cluster [clabel [i]].append (label [i])


    train_data = np.loadtxt (os.path.join(data_address, 'info_train_fold_%d.csv'%bootstrap_number), skiprows=1, usecols=[1])

    train_label = train_data
    filenames = pd.read_csv (os.path.join(data_address, 'info_train_fold_%d.csv'%bootstrap_number), usecols=[0], sep=' ')
    cluster_info = []
    for i, row in filenames.itertuples ():
            cluster_info.append ([row, clabel [i]])

    def clean_file(fname):
        if os.path.isfile(fname):
            os.remove(fname)
        return fname

    def list_to_df(info, header=['filename', 'label']):
        df = pd.DataFrame(info)
        df.columns = header
        return df
       
    def savecsv(info, mode, filepath):
        df = pd.DataFrame(info)
        df.to_csv(clean_file(filepath), sep=' ', index=None, header=['filename', 'label', 'context', 'participant'])
        print("save file", filepath)

    
    
    def savecsv(info, filepath):
        df = pd.DataFrame(info)
        df.to_csv(clean_file(filepath), sep=' ', index=None, header=['filename', 'label'])

    savecsv (cluster_info, os.path.join(data_address, 'info_cluster_fold_%d_'%bootstrap_number+'.csv'))
    

    
    info_cluster_train, info_cluster_test = train_test_split(cluster_info, test_size=0.2, random_state=random_seed)
    savecsv(info_cluster_train, os.path.join(data_address, "info_cluster_fold_%d_train.csv"%(bootstrap_number)))
    savecsv(info_cluster_test, os.path.join(data_address, "info_cluster_fold_%d_test.csv"%(bootstrap_number)))
    

    import pickle
    par_to_num = pickle.load(open(os.path.join(data_address, "par_to_num.dict"), "rb" ) )
    cluster_info_df = list_to_df(cluster_info)
        
    info_train, info_test = cluster_info_df[cluster_info_df['participant']!=par_to_num[bootstrap_number]], cluster_info_df[cluster_info_df['participant']==par_to_num[bootstrap_number]]
    savecsv(info_train, 'train', os.path.join(save_path, "info_train_fold_%d.csv"%bootstrap_number))
    savecsv(info_test, 'test', os.path.join(save_path, "info_test_fold_%d.csv"%bootstrap_number))


    #TODO: to be fixed
    del content, features, label, kmeans, clabel, label_in_cluster, train_data, train_label, filenames, cluster_info
'''


def train_beta (bootstrap_number, optimization_params, data_address, cache_address, results_address, CUDA_DEVICE, num_clusters=0):
    tfboard_dir = make_dir (
        os.path.join(results_address, "single_network_bootstrap_%d"%bootstrap_number))
    writer = SummaryWriter(log_dir=tfboard_dir)

    train_loader = torch.utils.data.DataLoader(dataset(data_address, data_address, {'train': "info_train_fold_%d.csv", 'test':"info_test_fold_%d.csv"}, True, bootstrap_number), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset(data_address, data_address, {'train': "info_train_fold_%d.csv", 'test':"info_test_fold_%d.csv"}, False, bootstrap_number), batch_size=64, shuffle=False)
    model = beta(num_clusters)
    model.cuda(CUDA_DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=optimization_params['lr'], momentum=optimization_params['momentum'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, threshold=0.02, cooldown=5)

    for epoch in range(1, optimization_params['epochs'] + 1):
        train_accr = train (model, writer, train_loader, optimizer, epoch, CUDA_DEVICE, results_address)
        val_accr = test (model, writer, test_loader, epoch, CUDA_DEVICE, results_address, bootstrap_number)
        scheduler.step (val_accr)
    
    torch.save (model.state_dict (), os.path.join(cache_address, 'pretrain_fold_%d.pt'%(bootstrap_number)))

    writer.close()
    del model, optimizer, scheduler


def train_alpha (num_clusters, optimization_params, bootstrap_number, data_address, cache_address, results_address, CUDA_DEVICE):
    tfboard_dir = make_dir (
        os.path.join(results_address, "bootstrap_%d"%bootstrap_number+"_%d_clusters"%num_clusters))
    writer = SummaryWriter(log_dir=tfboard_dir)

    train_loader = torch.utils.data.DataLoader(dataset(data_address, cache_address, {'train': "info_cluster_fold_%d_train.csv", 'test':"info_cluster_fold_%d_test.csv"}, True, bootstrap_number, True) , batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset(data_address, cache_address, {'train': "info_cluster_fold_%d_train.csv", 'test':"info_cluster_fold_%d_test.csv"} , False, bootstrap_number, True), batch_size=64, shuffle=False)

    model = alpha (num_clusters)
    model.cuda(CUDA_DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=optimization_params['lr'], momentum=optimization_params['momentum'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, threshold=0.02, cooldown=5)

    beginning_time = datetime.now()
    for epoch in range(1, optimization_params['epochs'] + 1):
        train_accr = train (model, writer, train_loader, optimizer, epoch, CUDA_DEVICE, results_address)
        val_accr = test (model, writer, test_loader, epoch, CUDA_DEVICE, results_address, bootstrap_number)
        scheduler.step (val_accr)

        time_diff = datetime.now() - beginning_time
        end_time = (optimization_params['epochs']/epoch) * time_diff + beginning_time

        print ('TRAIN SELECTOR END TIME:', end_time)

    torch.save (model.state_dict (), os.path.join(cache_address, 'selector_pretrain_fold_%d.pt'%(bootstrap_number)))

    writer.close()
    del model, optimizer, scheduler, train_loader


def extract (bootstrap_number, data_address, cache_address, CUDA_DEVICE):
    #print ('----------------------------'*5, CUDA_DEVICE, '----------'*5)
    train_loader = torch.utils.data.DataLoader (dataset (data_address, data_address, {'train': "info_train_fold_%d.csv", 'test':"info_test_fold_%d.csv"}, True, bootstrap_number), batch_size=1, shuffle=False)
    model = beta ()
    model.cuda (CUDA_DEVICE)
    model.load_state_dict (torch.load (os.path.join(cache_address, 'pretrain_fold_%d.pt'%(bootstrap_number))))

    extract_features (model, train_loader, bootstrap_number, cache_address, CUDA_DEVICE)
    del model
    del train_loader


def pretraining(num_clusters, bootstrap_number, results_address, data_address, cache_address, optimization_params, CUDA_DEVICE):

    log_display(mode='opening', data_dict = {'number of clusters':num_clusters, 'bootstrap number':bootstrap_number, 'current address': os.path.dirname(os.path.abspath(__file__)), 'results_address':results_address, 'data_address':data_address, 'cache_address':cache_address, 'CUDA device': CUDA_DEVICE, '# training epochs':optimization_params['epochs'], 'learning rate':optimization_params['lr'], 'momentum':optimization_params['momentum'], 'step size':optimization_params['step']}, results_address = results_address)

    #if os.path.isfile(os.path.join(cache_address, 'selector_pretrain_fold_%d.pt'%(bootstrap_number))):
    #    print ('pretraining skipped...')
    #    return
    
    log_display(mode='training_beta', results_address = results_address)

    train_beta (bootstrap_number, optimization_params, data_address, cache_address, results_address, CUDA_DEVICE, num_clusters)

    log_display(mode='cnn_extract', results_address = results_address)

    extract (bootstrap_number, data_address, cache_address, CUDA_DEVICE)

    log_display(mode='clustering', results_address = results_address)

    cluster (num_clusters, bootstrap_number, data_address, cache_address)

    log_display(mode='training_alpha', results_address = results_address)

    train_alpha (num_clusters, optimization_params, bootstrap_number, data_address, cache_address, results_address, CUDA_DEVICE)
