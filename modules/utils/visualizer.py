import numpy as np
import torch.nn.functional as F
from skimage.draw import disk

import matplotlib.pyplot as plt



class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]),self.kp_size, shape=image.shape[:2])
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

    def visualize(self,**kwargs):
        images = []

        if "source" in kwargs:
            # Source image with keypoints
            source = kwargs['source'].data.cpu().numpy()
            source = np.transpose(source, [0, 2, 3, 1])
            images.append(source)

        if 'compressed_source' in kwargs:
            comp_source = kwargs['compressed_source'].data.cpu().numpy()
            comp_source = np.transpose(comp_source, [0, 2, 3, 1])
            if "kp_source" in kwargs:
                kp_source = kwargs['kp_source']['value'].data.cpu().numpy()
                images.append((comp_source, kp_source))
            else:
                images.append(comp_source)

        if 'driving' in kwargs:
            # Driving image with keypoints
            driving = kwargs['driving'].data.cpu().numpy()
            driving = np.transpose(driving, [0, 2, 3, 1])
            if 'kp_driving' in kwargs:
                kp_driving = kwargs['kp_driving']['value'].data.cpu().numpy()
                images.append((driving, kp_driving))
            else:
                images.append(driving)
        
        if 'hevc' in kwargs:
            # hevc base layer
            hevc = kwargs['hevc'].data.cpu().numpy()
            hevc = np.transpose(hevc, [0, 2, 3, 1])
            images.append(hevc)
            
        ## Occlusion map
        if 'occlusion_map' in kwargs:
            occlusion_map = kwargs['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        if 'prediction' in kwargs:
            # Result with and without keypoints
            prediction = kwargs['prediction'].data.cpu().numpy()
            prediction = np.transpose(prediction, [0, 2, 3, 1])
            if 'kp_norm' in kwargs:
                kp_norm = kwargs['kp_norm']['value'].data.cpu().numpy()
                images.append((prediction, kp_norm))
            images.append(prediction)


        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
