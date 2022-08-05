import os
import numpy as np
import time
import math
import torch

import torch.nn.functional as F


from matplotlib import pyplot as plt
from tqdm import tqdm



def pc_from_txt(filename):
    """
    Loads a point cloud from a .txt file
    
    Parameters:
    filename:string - location for the .txt file
    
    Returns:
    point_cloud - numpy array of point cloud data, with points as rows (3 columns for 3d data)
    """
    point_cloud = np.loadtxt(filename)
    return point_cloud

def plot_downsample(point_cloud, sample_fraction = 0.1, random_seed=0, figsize=(10,10), is_scaled = True):
    """
    Plots a 3D point cloud using matplotlib
    
    Parameters:
    point_cloud - nx3 numpy array of the points to plot
    sample_fraction - downsampling fraction (float)
    random_seed - rng seed
    figsize - self explanatory (tuple)
    
    Returns:
    figure, axes for the plot
    """
    np.random.seed(random_seed)

    idx = np.random.randint(point_cloud.shape[0], size=int(point_cloud.shape[0]*sample_fraction))
    
    plot_points = point_cloud[idx,:]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    if is_scaled:
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))
    
    ax.scatter(plot_points[:,0], plot_points[:,1], plot_points[:,2], marker='.')
    
    return fig, ax

def center_and_scale(point_cloud):
    """
    Recenters a point cloud to the origin and
    scales to [-1,1]^3
    
    Parameters:
    point_cloud - nx3 numpy array of points
    
    Returns
    scaled_cloud - centered and scaled point cloud
    """
    centered_cloud = point_cloud-np.mean(point_cloud, axis=0)
    return centered_cloud/np.max(abs(centered_cloud))

def project_point_np(point, camera_projection, f=1):
    """
    Perspective projection of a single point on to image plane, using numpy.
    
    Parameters:
    point - 3D point to be projected as numpy array, in real world coordinates - Dimension 4 NOT 1x4 
    camera_projection - 4x4 camera projection matrix as numpy array:
                        x_camera_tilde = camera_projection * x_world_tilde (with x as col. vectors)
                        camera_transform = r11 r12 r13 Tx
                                           r21 r22 r23 Ty
                                           r31 r32 r33 Tz 
                                           0   0   0   1
                                           
    f - distance from camera to image plane. Default=1

    Returns
    projected_point - projected coordinates in image plane - (real world coordinates - not pixels)
    depth - depth of projected point (from camera)
    
    _tilde denotes homegeneous coordinates
    """ 
    
    x_world_tilde = np.append(point, 1)
    
    x_camera_tilde = np.matmul(camera_projection, x_world_tilde)
    

    
    perspective_projection = np.array([
                            [f, 0, 0, 0],
                            [0, f, 0, 0],
                            [0, 0, 1, 0],
                            ])
    
    x_image_tilde = np.matmul(perspective_projection, x_camera_tilde)
    
    x_image = np.array([x_image_tilde[0]/x_image_tilde[2], x_image_tilde[1]/x_image_tilde[2]])
    
    return x_image, x_camera_tilde[2] #x,y coordinates in image plane, depth

def project_points_np(points, camera_projection, f=1):
    """
    Projects a list/numpy array of points in one go, using numpy.
    """
    projected_points_depths = [project_point_np(point, camera_projection, f) for point in points]
    projected_points = np.array([x[0] for x in projected_points_depths])
    projected_depths = np.array([x[1] for x in projected_points_depths])
    
    return projected_points, projected_depths

def project_points(cloud, camera_projection, f=1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    if type(cloud) == np.ndarray:
        cloud = torch.from_numpy(cloud)
    
    x_world_tilde=torch.cat((cloud.clone().detach(), torch.ones(cloud.shape[0],1)), 1).transpose(0,1)  
    x_world_tilde=x_world_tilde.float()
    x_world_tilde=x_world_tilde.to(device=device)
    
    camera_projection = torch.tensor(camera_projection,
                                     dtype=torch.float32)
    
    perspective_projection = torch.tensor([
                            [f, 0, 0, 0],
                            [0, f, 0, 0],
                            [0, 0, 1, 0],
                            ], dtype=torch.float32)
    
    projection_matrix = torch.matmul(perspective_projection, camera_projection)
    projection_matrix = projection_matrix.to(device=device)
    
    x_image_tilde = torch.matmul(projection_matrix, x_world_tilde)
    
    return (x_image_tilde[0:2]/x_image_tilde[2]).transpose(0,1), x_image_tilde[2]


def depth_image_from_projection(points, depths, camera_fov_deg=90, image_dim=128, k=50, use_hard_min=False):
    """
    Plots a depth image from perspective projected points
    
    Parameters:
    points - nx2 numpy array of projected points
    depths - corresponding point depths from camera
    camera_fov_deg - camera FOV in degrees
    """
    
    #The maximum projected image coordinate based on camera FOV
    max_value = np.tan(np.radians(camera_fov_deg/2))
  
    no_points = len(points)
    points = points + max_value #Change range from -max->max to 0->2*max
    pixel_coords = torch.round(points/(2*max_value) * image_dim).long()

            
    if use_hard_min:
        max_coord = max(int(torch.max(pixel_coords)), image_dim)
        image = torch.zeros(size=(max_coord+1, max_coord+1))   

        for coord, depth in zip(pixel_coords, depths):
            i, j = int(coord[1]), int(coord[0])
            if (0 <= i < image_dim) and (0 <= j < image_dim): #If not outside the FOV
                if not(image[i, j]):
                    image[i, j] = depth
                elif depth < image[i, j]: #If there's something else there
                    image[i, j] = depth
                    
        return image
    
    else:
        pos_filter = torch.logical_and(pixel_coords[:,0]>=0, pixel_coords[:,1]>=0)
        max_filter = torch.logical_and(pixel_coords[:,0]<image_dim, pixel_coords[:,1]<image_dim)
        
        range_filter = torch.logical_and(pos_filter, max_filter)
        
        pixel_coords = pixel_coords[range_filter]
        depths = depths[range_filter]
        sparse = torch.sparse_coo_tensor(pixel_coords.t(), torch.pow(depths, -k), (image_dim, image_dim))
        sparse = sparse.coalesce()
        sparse = torch.pow(sparse, -1/k)
        return sparse.t().to_dense().to('cpu')#image[0:image_dim, 0:image_dim]

    
    

def get_depth_images_from_cloud(points, camera_fov_deg=90, image_dim=128, f=1, camera_dist=1.4, k=50, use_hard_min=False):
    """
    Returns a set of 6 depth images from axis viewpoints, given the point cloud (Not the projected points)
    """
    camera_projections = [
    #Front to back
    np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Back to front
    np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Bottom up
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Top down
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Left to right
    np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Right to left
    np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, camera_dist],
        [0, 0, 0, 1]
    ])         
    ]
    
    start = time.time()
    
    points_depths = [project_points(cloud=points, 
                                    camera_projection=camera_projection, 
                                    f=f) 
                     for camera_projection in camera_projections]
    
    end = time.time()
    #print("Time projecting: ", end-start)
    
    start=time.time()

    depth_images = [depth_image_from_projection(points=points, 
                                                depths=depths, 
                                                camera_fov_deg=camera_fov_deg, 
                                                image_dim=image_dim, 
                                                k=k,
                                                use_hard_min=use_hard_min).unsqueeze(axis=0) 
                    for points, depths in points_depths]
    
    depth_images = torch.cat(depth_images)
    end=time.time()
    #print("Time imaging: ", end-start)
    
    
    
    return depth_images

def plot_depth_image(depth_image):
    """
    Plots a single depth image using the matplotlib OO interface
    """
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(depth_image, origin='lower')
    
    return fig, ax
    
def plot_depth_images(depth_images, nrows=2, figsize=None):
    """
    Plots a full set of projections (6 as in the paper) across 2 rows
    using the matplotlib OO interface
    """
    from_dataset = False
    if len(depth_images.shape) == 4:
        from_dataset = True
    
    no_images = len(depth_images)
    
    ncols = math.ceil(no_images/nrows)
    
    if figsize == None:
        figsize = (5*ncols, 5*nrows)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    
    for i in range(no_images):
        if from_dataset:
            if nrows>1:
                ax[i//ncols,i%ncols].imshow(depth_images[i][0], origin='lower')
            else:
                ax[i].imshow(depth_images[i][0], origin='lower')
        else:
            if nrows>1:
                ax[i//ncols,i%ncols].imshow(depth_images[i], origin='lower')
            else:
                ax[i].imshow(depth_images[i], origin='lower')
        
    return fig, ax    

def smooth_loss(pred, gold, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed.
    This isn't used any more, since torch now supports label smoothing."""

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
