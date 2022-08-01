import utils
import os
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

from tqdm import tqdm
from torch.utils.data import Dataset

import pandas as pd

class TreeSpeciesPointDataset(Dataset):
    """Dataset for tree species classification from
    point cloud depth projection images"""

    def __init__(self, data_dir, metadata_file):
        """
        Args:
            metadata_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
        """
        meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
        self.species = list(meta_frame.groupby('sp')['id'].nunique().keys())
        
        self.counts = meta_frame['sp'].value_counts() 
        
        self.data_dir = data_dir
        
        self.point_clouds = []
        self.labels = None
        self.file_names = None
        self.meta_frame = pd.DataFrame().reindex_like(meta_frame) #A new dataframe that will only contain the subset of examples from the original that are found in the data directory

        
        self.image_dim = None
        self.camera_fov_deg = None
        self.f = None
        self.camera_dist = None
        self.transforms = None
        
        self.min_rotation = None
        self.max_rotation = None
        
        self.min_translation = None
        self.max_translation = None
        
        self.min_scale = None
        self.max_scale = None
        
        
        filenames = list(filter(lambda t:t.endswith('.txt'), os.listdir(self.data_dir)))
        no_files = len(filenames)
        
        self.labels = torch.zeros(no_files)
        
        for i, file in tqdm(enumerate(filenames), total=no_files): #For each file in the directory
            file_name = self.data_dir + file
            self.file_names.append(file_name) #Save the file name
            cloud = utils.pc_from_txt(file_name) #Load the point cloud
            cloud = utils.center_and_scale(cloud) #Center and scale it
            
            self.point_clouds.append(torch.from_numpy(cloud)) #Add the point cloud to the dataset 
            
            meta_entry = meta_frame[meta_frame.id==file[:-4]] #Get the relevant entry from the dataframe for this filename
            self.labels[i] = self.species.index(meta_entry.sp.values[0]) #Add the species label (int, index from self.species) to the list of labels

            self.meta_frame = self.meta_frame.append(meta_entry) #Add to new meta frame
            
        self.labels = self.labels.long()
        
        return
    
    
    def get_depth_image(self, i, transforms = None):
        if transforms is None:
            transforms = self.transforms
        
        points = self.point_clouds[i]    
            
        if 'rotation' in transforms:
            points = self.random_rotation(points, 
                                          min_rotation=self.min_rotation,
                                          max_rotation=self.max_rotation)
            
        if 'translation' in transforms:
            points = self.random_translation(points,
                                             min_translation=self.min_translation,
                                             max_translation=self.max_translation)
            
        if 'scaling' in transforms:
            points = self.random_scaling(points,
                                         min_scale=self.min_scale,
                                         max_scale=self.max_scale)
            
        
        return torch.unsqueeze(
               utils.get_depth_images_from_cloud(points=points, 
                                                 image_dim=self.image_dim, 
                                                 camera_fov_deg=self.camera_fov_deg, 
                                                 f=self.f, 
                                                 camera_dist=self.camera_dist
                                                 )
                                    , 1)
    
    
    def remove_species(self, species):
        
        idx = [] #Indices to keep
        
        for i in range(len(self.labels)): #Remove entries in images and labels for that species
            if not(self.species[int(self.labels[i])] == species):
                idx.append(i)
                
        self.point_clouds = [self.point_clouds[i] for i in idx] #Crop point clouds
        self.labels = self.labels[idx] #Crop labels
        self.meta_frame = self.meta_frame.iloc[idx] #Crop meta frame
            
        old_species = self.species.copy() 
        self.species.pop(self.species.index(species)) #Pop from species list
        
            
        species_map = [self.species.index(species) if species in self.species else None for species in old_species]     
            

        for k in range(len(self.labels)): #Apply species map to relabel
            self.labels[k] = torch.tensor(species_map[int(self.labels[k])])
            
        self.counts = self.counts.drop(species)
        
        return
    
    def set_params(self, 
                   image_dim = None,
                   camera_fov_deg = None,
                   f = None,
                   camera_dist = None,
                   transforms = None,
                   min_rotation = None,
                   max_rotation = None,
                   min_translation = None,
                   max_translation = None,
                   min_scale = None,
                   max_scale = None):
     
        if image_dim:    
            self.image_dim = image_dim        
        if camera_fov_deg:
            self.camera_fov_deg = camera_fov_deg 
        if f:
            self.f = f      
        if camera_dist:
            self.camera_dist = camera_dist      
        if transforms:
            self.transforms = transforms
            
            if 'rotation' in transforms:
                self.min_rotation = min_rotation
                self.max_rotation = max_rotation
                
            if 'translation' in transforms:
                self.min_translation = min_translation
                self.max_translation = max_translation
                
            if 'scaling' in transforms:
                self.min_scale = min_scale
                self.max_scale = max_scale
            
        return
    
    def random_rotation(self,
                        point_cloud,
                        min_rotation=0,
                        max_rotation=2*torch.pi):
          
        theta = torch.rand(1)*(max_rotation - min_rotation) + min_rotation
        
        Rz = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0               ,                 0, 1],
        ]).double()
        
        return torch.matmul(point_cloud, Rz.t())
    
    def random_translation(self,
                           point_cloud,
                           min_translation = 0,
                           max_translation = 0.1):
        
        sign = torch.sign(torch.rand(1) - 0.5)
        tran = torch.rand(3)*(max_translation - min_translation) + min_translation
        
        return point_cloud + sign*tran
    
    def random_scaling(self,
                       point_cloud,
                       min_scale = 0.5,
                       max_scale = 1.5):
        
        
        scale = torch.rand(1)*(max_scale - min_scale) + min_scale
        
        return scale * point_cloud
    
    def __len__(self):
        assert len(self.labels) == len(self.point_clouds)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            num_trees = len(idx)
        elif type(idx) == int:
            num_trees = 1
        

        depth_images = torch.zeros(size=(num_trees, 6, 1, self.image_dim, self.image_dim))
        
        if type(idx) == list:
            for i in range(len(idx)):
                depth_images[i] = self.get_depth_image(int(idx[i]))
        elif type(idx) == int:
            depth_images = self.get_depth_image(idx)
        
        labels = self.labels[idx]
        
        sample = {'depth_images': depth_images, 'labels': labels}

        return sample    