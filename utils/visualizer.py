import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import disk

import matplotlib.pyplot as plt
import collections



class Visualizer:
    def __init__(self, kp_size=5, draw_border=True, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)
        
    def kp_view(self,source, kp_source):
    	images = []
    	src = source.data.cpu().numpy()
    	image = np.transpose(src, [0,2,3,1])
    	kp = kp_source['value'].data.cpu().numpy()
    	images.append((image, kp))
    	image = self.create_image_grid(*images)
    	image = (255 * image).astype(np.uint8)
    	return image

    def visualize(self, driving, source, out):
        images = []
        # Source image with keypoints
        source = source.data.cpu().numpy()
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        
        blank = np.zeros(driving.shape)
        blank[:] = 1
        images.append((blank, kp_driving))
        
        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
