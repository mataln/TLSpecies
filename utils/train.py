import os
import sys
import copy
from tkinter import wantobjects
pdir = os.path.dirname(os.getcwd())
sys.path.append(pdir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import random

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import confusion_matrix
from datetime import datetime

import utils
from simpleview_pytorch import SimpleView

from torch.utils.data.dataset import Dataset


def train(train_data, val_data, test_data, model_dir, params, fname_prefix = str(datetime.now()), wandb_project=None, init_wandb=True):
    """
    Trains a model

    train_file - location of train data file
    val_data - location of validation data file
    test_data - location of test data file

    saves trained model to disk at the folder model_dir/
    params to specify training parameters
    fname_prefix to determine model filenames - fname_prefix_best_test for best overall test accuracy etc.
    optional wandb logging by setting name of wandb project - should prompt for login automatically.
    """
    #wandb.login()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = torch.load(train_data)
    val_data = torch.load(val_data)
    test_data = torch.load(test_data)

    val_data.set_params(transforms = ["none"]) #Turn off transforms for validation, test sets.
    test_data.set_params(transforms = ["none"])      
    
    print('Training data:')
    print(train_data.counts)
    print('Species: ', train_data.species)
    print('Labels: ', train_data.labels)
    print('Total count: ', len(train_data))

    print('Validation data:')
    print(val_data.counts)
    print('Species: ', val_data.species)
    print('Labels: ', val_data.labels)
    print('Total count: ', len(val_data))

    print('Test data:')
    print(test_data.counts)
    print('Species: ', test_data.species)
    print('Labels: ', test_data.labels)
    print('Total count: ', len(test_data))


                  

    if wandb_project:
        experiment_name = wandb.util.generate_id()

        if init_wandb:
            run = wandb.init(
                project=wandb_project,
                group=experiment_name,
                config=params,
                reinit=False
                )
        
        wandb.config.update(params)
        config = wandb.config


    torch.manual_seed(params['random_seed'])
    torch.cuda.manual_seed(params['random_seed'])
    torch.random.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])

    for specie in list(set(train_data.species) - set(params['species'])):
        print("Removing: {}".format(specie))
        train_data.remove_species(specie)
        val_data.remove_species(specie)
        test_data.remove_species(specie)

    assert train_data.species == val_data.species, "Warning - train/val set species labels do not match. Check content and order"
    assert train_data.species == test_data.species, "Warning - train/test set species labels do not match. Check content and order"

    print('Train Dataset:')
    print(train_data.counts)
    print('Species: ', train_data.species)
    print('Labels: ', train_data.labels)
    print('Total count: ', len(train_data))
    print()

    print('Validation Dataset:')
    print(val_data.counts)
    print('Species: ', val_data.species)
    print('Labels: ', val_data.labels)
    print('Total count: ', len(val_data))
    print()

    print('Test Dataset:')
    print(test_data.counts)
    print('Species: ', test_data.species)
    print('Labels: ', test_data.labels)
    print('Total count: ', len(test_data))
    print()

    assert set(params['species']) == set(train_data.species)


    #Train sampler==========================================
    if params['train_sampler'] == "random": 
        print("Using random/uniform sampling...")
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], 
                                                    shuffle=params["shuffle_dataset"]) #Dataset without a sampler (uniform/deterministic if shuffled/unshuffled)

    elif params['train_sampler'] == "balanced":
        print("Using balanced sampling...")
        labels = train_data.labels #Counts over 
        counts = torch.bincount(labels) #Training set only
        label_weights = 1 / counts 

        sample_weights = torch.stack([label_weights[label] for label in train_data.labels]) #Corresponding weight for each sample

        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) #Replacement is true by default for the weighted sampler
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], 
                                                    sampler=train_sampler) #Dataloader using the weighted sampler
    #=======================================================    



    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size']) #Val loader - never shuffled (shouldn't matter anyway)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['batch_size']) #Test loader - never shuffled
                                                

    if params['model']=="SimpleView":
        model = SimpleView(
            num_views=6,
            num_classes=len(params['species'])
        )

    model = model.to(device=device)

    if params['loss_fn']=="cross-entropy":
        loss_fn = nn.CrossEntropyLoss()
        print("Using cross-entropy loss...")
    if params['loss_fn']=="smooth-loss":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
        print("Using smooth-loss")

    if type(params['learning_rate']) == list:
        lr = params['learning_rate'][0]
        step_size = params['learning_rate'][1]
        gamma = params['learning_rate'][2]
    else:
        lr = params['learning_rate']

    if params['optimizer']=="sgd":
        print("Optimizing with SGD...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=params['momentum'])
    elif params['optimizer']=="adam":
        print("Optimizing with AdaM...")
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if type(params['learning_rate']) == list:
        print("Using step LR scheduler...")
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    #wandb.watch(model)
    best_acc = 0
    best_min_acc = 0
    best_avg_acc = 0

    best_test_acc = 0
    best_avg_test_acc = 0
    best_min_test_acc = 0
    
    for epoch in range(params['epochs']):  # loop over the dataset multiple times

        #Training loop============================================
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            depth_images = data['depth_images']
            labels = data['labels']

            depth_images = depth_images.to(device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(depth_images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        #Test loop================================================
        num_train_correct = 0
        num_train_samples = 0

        num_val_correct = 0
        num_val_samples = 0
        
        num_test_correct = 0
        num_test_samples = 0

        running_train_loss = 0
        running_val_loss = 0
        running_test_loss = 0

        model.eval()  
        with torch.no_grad():
            #Train set eval==============
            for data in train_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)
                num_train_correct += (predictions == labels).sum()
                num_train_samples += predictions.size(0)

                running_train_loss += loss_fn(scores, labels)

            train_acc = float(num_train_correct)/float(num_train_samples)
            train_loss = running_train_loss/len(train_loader)

            
            
            
            

            #Val set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in val_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_val_correct += (predictions == labels).sum()
                num_val_samples += predictions.size(0)

                running_val_loss += loss_fn(scores, labels)

            val_acc = float(num_val_correct)/float(num_val_samples)
            val_loss = running_val_loss/len(val_loader)

            print(f'OVERALL (Val): Got {num_val_correct} / {num_val_samples} with accuracy {val_acc*100:.2f}')

            cm = confusion_matrix(all_labels.cpu(), all_predictions.cpu())
            totals = cm.sum(axis=1)
            
            accs = np.zeros(len(totals))
            for i in range(len(totals)):
                accs[i] = cm[i,i]/totals[i]
                print(f"{train_data.species[i]}: Got {cm[i,i]}/{totals[i]} with accuracy {(cm[i,i]/totals[i])*100:.2f}")
                if wandb_project:
                    wandb.log({f"{train_data.species[i]} Accuracy":(cm[i,i]/totals[i])}, commit = False)


            if val_acc >= best_acc:
                best_model_state = copy.deepcopy(model.state_dict())
                best_acc = val_acc

            if wandb_project:    
                wandb.log({"Best_acc":best_acc}, commit = False)
                
            if min(accs) >= best_min_acc:
                best_min_model_state = copy.deepcopy(model.state_dict())
                best_min_acc = min(accs)

            avg_acc = sum(accs)/len(accs)
            if avg_acc > best_avg_acc:
                best_avg_model_state = copy.deepcopy(model.state_dict())
                best_avg_acc = avg_acc

            if wandb_project:    
                wandb.log({"Best_min_acc":best_min_acc}, commit = False)
            #==================================
            
            
            
            
            
            
            
            
            
            #test set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in test_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_test_correct += (predictions == labels).sum()
                num_test_samples += predictions.size(0)

                running_test_loss += loss_fn(scores, labels)

            test_acc = float(num_test_correct)/float(num_test_samples)
            test_loss = running_test_loss/len(test_loader)

            print(f'OVERALL (test): Got {num_test_correct} / {num_test_samples} with accuracy {test_acc*100:.2f}')

            cm = confusion_matrix(all_labels.cpu(), all_predictions.cpu())
            totals = cm.sum(axis=1)
            
            accs = np.zeros(len(totals))
            for i in range(len(totals)):
                accs[i] = cm[i,i]/totals[i]
                print(f"{train_data.species[i]}: Got {cm[i,i]}/{totals[i]} with accuracy {(cm[i,i]/totals[i])*100:.2f}")
                if wandb_project:
                    wandb.log({f"{train_data.species[i]} Accuracy":(cm[i,i]/totals[i])}, commit = False)


            if test_acc >= best_test_acc:
                best_test_model_state = copy.deepcopy(model.state_dict())
                best_test_acc = test_acc

            if wandb_project:    
              wandb.log({"Best_test_acc":best_test_acc}, commit = False)
                
            if min(accs) >= best_min_test_acc:
                best_min_test_model_state = copy.deepcopy(model.state_dict())
                best_min_test_acc = min(accs)

            avg_test_acc = sum(accs)/len(accs)
            if avg_test_acc > best_avg_test_acc:
                best_avg_test_model_state = copy.deepcopy(model.state_dict())
                best_avg_test_acc = avg_test_acc

            if wandb_project:    
                wandb.log({"Best_min_test_acc":best_min_test_acc}, commit = False)
            #==================================
            
            
            
            
            if wandb_project:    
                wandb.log({
                    "Train Loss":train_loss,
                    "Validation Loss":val_loss,
                    "Test Loss":test_loss,
                    "Train Accuracy":train_acc,
                    "Validation Accuracy":val_acc,
                    "Test Accuracy":test_acc,
                    "Learning Rate":optimizer.param_groups[0]['lr'],
                    "Epoch":epoch
                    })

            scheduler.step()


    print('Finished Training')

    if not(os.path.exists(model_dir)):
        print("Creating model directory...")
        os.makedirs(model_dir)

    if wandb_project:
        fname = wandb.run.name

    print('Saving best (val) model...')
    print('Best overall accuracy: {}'.format(best_acc))
    torch.save(best_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_best')
              )
    print('Saved!')
    
    print('Saving converged model...')
    print('Converged accuracy: {}'.format(val_acc))
    converged_model_state = copy.deepcopy(model.state_dict())
    torch.save(converged_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_converged')
              )
    print('Saved!')
    
    print('Saving best producer (val) accuracy model...')
    print('Best min producer accuracy: {}'.format(best_min_acc))
    torch.save(best_min_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_best_prod')
              )
    print('Saved!')

    print('Saving best average (val) accuracy model...')
    print('Best average producer accuracy: {}'.format(best_avg_acc))
    torch.save(best_avg_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_best_avg')
              )
    print('Saved!')
    
    print('Saving best (test) model...')
    print('Best overall (test) accuracy: {}'.format(best_test_acc))
    torch.save(best_test_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_best_test'))
               
    print('Saving best (test) producer accuracy model...')
    print('Best (test) min producer accuracy: {}'.format(best_min_test_acc))
    torch.save(best_min_test_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_best_test_prod'))

    print('Saving best average (test) accuracy model...')
    print('Best average producer accuracy: {}'.format(best_avg_test_acc))
    torch.save(best_avg_test_model_state,
               '{model_dir}/{fname}'.format(
                   model_dir=model_dir,
                   fname=fname_prefix+'_best_avg_test')
              )
    print('Saved!')
               


    if "run" in locals():
        run.finish()