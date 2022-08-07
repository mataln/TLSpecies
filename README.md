# TreeLS
Implementation for TLS species ID paper - placeholder.

**(IMPORTANT) A note before this readme begins** - the code for this paper was written when the original publication, 
*Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline* (Goyal et al., 2021), was under review and code from the original authors 
was not available. Now that it has been published, you can find it [here](https://github.com/princeton-vl/SimpleView). You might find that it is better 
optimised, or in some way more flexible/easier to use. Otherwise, details for this repository can be found below. <br/>
<br/>
==================================================================================<br/>

![image](projections.png)

[intro]

## Folder Structure
An outline of the layout of this repository can be seen below, including where data is expected to be stored, and in what format. Our data, and metadata can be found [here](https://zenodo.org/record/6962717#.Yu_Dc_HMK3I) (DOI:10.5281/zenodo.6962717; originally published in Owen et al. 2021).


```
|-- LICENSE 
|-- README.md 
|-- TreeLS.yml 
| 
|-- data 
|   |-- treesXYZ 
|       |-- tree_id1.txt --> .txt files with containing point cloud data 
|                             i.e x1 y1 z1
|                                 x2 y2 z2
|       |-- tree_id2.txt      
|       |-- ... 
|
|   |-- meta
|       |-- tree-meta.csv --> metadata file describing species for each sample in treesXYZ
|                             it should have two columns 'id' and 'sp' containing identifiers and species labels
|                             with the id matching the filename for the corresponding pointcloud (w/o file extension)
|
|                             e.g. 
|                             id        sp
|                             tree_id1  QUEFAG
|                             ...
| 
|-- utils 
|   |-- __init__.py 
|   |-- dataset.py --> Custom pytorch dataset, including data augmentation
|   |-- utils.py --> Various utilities not directly related to training, plotting & preprocessing etc.
|   |-- train.py --> Training loop
|   |-- test.py --> Test loop
| 
|-- sh 
|   |-- dl-simpleview.sh --> Pulls simpleview model code from skeleton repo (https://github.com/isaaccorley/simpleview-pytorch; which was available during review)
```

## Demonstration notebook
A demonstration of the entire pipeline from raw data to the forward pass can be seen in demo.ipnyb, including the projection process.

## Building PyTorch datasets
The custom dataset class used to perform both 6-way perspective projection and data augmentation can be found in utils/dataset.py, and its use is demonstrated in demo.ipynb. Note that dataset is loaded in its entirety in one go, and lazy loading for very large datasets is not supported. Transforms will be forced off if performing the forward pass on an entire dataset.

## Inference
The forward pass for both an entire dataset and an individual sample are demonstrated in demo.ipynb

## Bibliography
*Revisiting point cloud classification with a simple and effective baseline*, Goyal et al., 2021 <br/>
*Competitive drivers of interspecific deviations of crown morphology from theoretical predictions measured with Terrestrial Laser Scanning*, Owen et al. 2021
