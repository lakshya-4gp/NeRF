import os
import numpy as np
import glob
from PIL import Image
import math
from collections import defaultdict
from pytorch3d.renderer import (
    FoVPerspectiveCameras
)
import torch
import random

def my_deterministic_shuffle(files):
    files.sort()
    random.Random(41).shuffle(files)
    return files

class MyCam:
    def __init__(self, H, W, intr, c2ws):
        # c2w: List of c2ws
        self.H, self.W, self.intr, self.c2ws = H, W, intr, c2ws
    def get(self,):
        return self.H, self.W, self.intr, self.c2ws

def get_data(train_not_test=True, num_files = 40, resize_ratio = 1.0, path = "bottles/", convert_to_opengl_convention = True):
    def get_mat(file):
        with open(file, "r") as f:
            return np.array([[float(k) for k in line.strip().split(" ")] for line in f.readlines() if line.strip() != ""])
    
    # Get Intrinsic Matrix
    intr = get_mat(os.path.join(path, "intrinsics.txt"))

    # for this dataset size = (800,800)
    size = (800,800)
    resized_size= (int(size[0]*resize_ratio), int(size[1]*resize_ratio))

    # resize intrinsic
    intr[:2,:] *= resize_ratio

    # datastructure to store data
    train, test = {'file':[], 'img':[], 'img_silhote':[], 'c2w':[]}, {'file':[], 'c2w':[]}

    # Take num_files from shuffled files.
    for f in my_deterministic_shuffle(glob.glob(os.path.join(path, "pose/*"))):
        # poses show the c2w matrices
        c2w = get_mat(f)
        # multiply columns as this is c2w, which is equivalent to multiply rows in w2c
        if convert_to_opengl_convention:
            c2w[:, 1:3] *= -1

        # if training_samples to be collected
        if train_not_test and ('train' in f or 'val' in f):
            # Get the image, resize it, convert to numpy array
            img = Image.open(f.replace("pose","rgb").replace("txt",'png'))
            assert img.size == size
            img = img.resize(resized_size)
            img = np.array(img.convert('RGB').getdata(), dtype=np.float32).reshape((*resized_size,3))
            # get silhoute by simply checking not a white background
            img_silhote = ((img != 255).sum(axis=-1) > 0).astype(np.float32)
            #normalize img
            img = img / 255
            train['file'].append(f)
            train['img'].append(img)
            train['img_silhote'].append(img_silhote)
            train['c2w'].append(c2w)
        # if test samples to be collected
        elif not train_not_test and ('test' in f):
            test['file'].append(f)
            test['c2w'].append(c2w)
        
        # if we have required num_files, break
        if len(test['file']) >= num_files or len(train['file']) >= num_files:
            break

    # Return aptly
    if train_not_test:
        return train['file'], MyCam(*resized_size, intr, train['c2w']), np.array(train['img']), np.array(train['img_silhote'])
    else:
        return test['file'], MyCam(*resized_size, intr, test['c2w'])




## ======================================================= For pytorch3d tutorial ====================================================

# for us resizing doesn't matter
def get_FOV_c2ws(size, ext, intr):
    w, h = size
    fx, fy = intr[0,0], intr[1,1]
    assert w == h and fx == fy
    fov_radians = 2 * math.atan(h/(2*fy))
    fov_deg = fov_radians * (180 / math.pi)
    aspect_ratio = 1
    znear = 0.1
    zfar = 100
    R = ext[:3,:3]
    T = ext[:3,3]
    return R, T, znear, zfar,  aspect_ratio, fov_deg

def getfovcam(params):
    R = torch.tensor([p[0] for p in params], dtype=torch.float32)
    T = torch.tensor([p[1] for p in params], dtype=torch.float32)
    znear = torch.tensor([p[2] for p in params], dtype=torch.float32)
    zfar = torch.tensor([p[3] for p in params], dtype=torch.float32)
    aspect_ratio = torch.tensor([p[4] for p in params], dtype=torch.float32)
    fov = torch.tensor([p[5] for p in params], dtype=torch.float32)
    return FoVPerspectiveCameras(
        R = R, 
        T = T, 
        znear = znear,
        zfar = zfar,
        aspect_ratio = aspect_ratio,
        fov = fov,
        device = 'cpu'
    )

def get_cam_img(train_not_test=True, num_files = 40, resize_ratio = 1.0, path = "bottles/", convert_to_pytorch3d_convention = False):
    def get_mat(file):
        with open(file, "r") as f:
            return np.array([[float(k) for k in line.strip().split(" ")] for line in f.readlines() if line.strip() != ""])
    
    # Get Intrinsic Matrix
    intr = get_mat(os.path.join(path, "intrinsics.txt"))

    # for this dataset size = (800,800)
    size = (800,800)
    resized_size= (int(size[0]*resize_ratio), int(size[1]*resize_ratio))
    # resize intrinsic
    intr[:2,:] *= resize_ratio

    # datastructure to store data
    train, test = {'file':[], 'img':[], 'img_silhote':[], 'c2w':[]}, {'file':[], 'c2w':[]}

    # Take num_files from shuffled files.
    for f in my_deterministic_shuffle(glob.glob(os.path.join(path, "pose/*"))):
        # get extrinsic matrix and use it to get the camera params: R,T,znear, zfar, aspect_ratio, fov
        ext = get_mat(f)
        # take inverse as the camera2world matrix is provided and FOVPerspective uses world2camera
        ext = np.linalg.inv(ext)
        # Camera follows the open cv convention(x right, y down, z forward), to pytorch3d convention (x left, y up, z forward)
        if convert_to_pytorch3d_convention:
            ext[0:2, :] *= -1
        c2w = get_FOV_c2ws(size, ext, intr)

        # if training_samples to be collected
        if train_not_test and ('train' in f or 'val' in f):
            # Get the image, resize it, convert to numpy array
            img = Image.open(f.replace("pose","rgb").replace("txt",'png'))
            assert img.size == size
            img = img.resize(resized_size)
            img = np.array(img.convert('RGB').getdata()).reshape((*resized_size,3))
            # get silhoute by simply checking not a white background
            img_silhote = (img != 255).sum(axis=-1) > 0
            #normalize img
            img = img / 255
            train['file'].append(f)
            train['img'].append(img)
            train['img_silhote'].append(img_silhote)
            train['c2w'].append(c2w)
        # if test samples to be collected
        elif not train_not_test and ('test' in f):
            test['file'].append(f)
            test['c2w'].append(c2w)
        
        # if we have required num_files, break
        if len(test['file']) >= num_files or len(train['file']) >= num_files:
            break

    # Return aptly
    if train_not_test:
        return train['file'], getfovcam(train['c2w']), torch.tensor(train['img'], dtype=torch.float32), torch.tensor(train['img_silhote'], dtype=torch.float32)
    else:
        return test['file'], getfovcam(test['c2w'])


if __name__ == '__main__':
    get_cam_img()
