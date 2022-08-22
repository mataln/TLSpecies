import os
import sys
import copy
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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR


import utils
from simpleview_pytorch import SimpleView

from torch.utils.data.dataset import Dataset

def predict(dataset, model, params):
    '''
    Given a dataset, test indices (n<N indices)
    and a trained model, returns:
    
    logits, labels, predictions
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for specie in list(set(dataset.species) - set(params["species"])):
        print("Removing: {}".format(specie))
        dataset.remove_species(specie)
            
    dataset.set_params(transforms=['none'])
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    

    model = model.to(device=device)
    model.eval()
    
    with torch.no_grad():
        
        all_logits = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
        all_predictions = torch.tensor([]).to(device)
        
        for data in loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)
                
                all_logits = torch.cat((all_logits, scores))
                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                
        return all_logits, all_labels, all_predictions
    
def predict_from_dirs(dataset_dir, model_dir, params):#Load data
    dataset = torch.load(dataset_dir)

    #Load model
    model = SimpleView(
            num_views=params["num_views"],
            num_classes=len(params["species"])
            )

    model.load_state_dict(torch.load(model_dir))

    logits, labels, predictions = utils.predict(dataset=dataset, model=model, params=params)
    
    return logits, labels, predictions, dataset.species

