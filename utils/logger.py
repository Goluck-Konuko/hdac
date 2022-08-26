
import os
import torch
import imageio
import torch.nn.functional as F
from skimage.draw import disk
import numpy as np
from typing import Dict,List,Tuple, Any
import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir: str, checkpoint_freq: int=100, 
                        visualizer_params: Dict[str, Any]=None, 
                        zfill_num: int = 8, log_file_name: str='log.txt', 
                        mode: str='test') -> None:
        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.mode = mode
        self.epoch_losses = []
        self.rec_video = []

    def log_scores(self, loss_names: List[str]) -> None:
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        self.epoch_losses = {y[0]: float(y[1]) for y in [x.split('-') for x in loss_string.split(';')]}

        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp: Dict[str, Any], out: Dict[str, Any], name: str=None) -> np.array:
        if name:
            visualizations_dir =self.visualizations_dir+f'_{name}' 
        else:
            visualizations_dir = self.visualizations_dir
        
        if not os.path.exists(visualizations_dir):
            os.makedirs(visualizations_dir)

        image = self.visualizer.visualize(inp,out)
        imageio.imsave(os.path.join(visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)
        
        return image

    def save_cpk(self, emergent: bool=False) -> None:
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-new-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path: str, generator: Any = None, discriminator : Any =None, 
                 kp_detector : Any =None, optimizer_generator: Any =None, 
                 optimizer_discriminator: Any=None, optimizer_kp_detector: Any=None) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            try:
                optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            except ValueError as e:
                print("Optimizer is randomly initialized")
                optimizer_generator.state_dict()
                
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except Exception as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
                optimizer_discriminator.state_dict()
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses: torch.tensor):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch: int, models: Dict[str, Any]=None,
                         inp: Dict[str, Any]=None, out: Dict[str, Any]=None, 
                         name: str = None, save: bool=False) -> tuple:
        self.epoch = epoch
        if models is not None:
            self.models = models
            if (self.epoch + 1) % self.checkpoint_freq == 0:
                self.save_cpk()
        if len(self.loss_list) > 0:
            self.log_scores(self.names)
        image= self.visualize_rec(inp, out, name=name)
        if save:
            self.rec_video.append(image)
        return image, self.epoch_losses


class Visualizer:
    def __init__(self, kp_size: int = 5, draw_border: bool = False, 
                        colormap: str='gist_rainbow') -> None:
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image: np.array, kp_array: np.array) -> np.array:
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]),self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images: np.array, kp: np.array)-> np.array:
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images: np.array) -> np.array:
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args) -> np.array:
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, inp: Dict[str, Any],out: Dict[str, Any]=None):
        images = []
        # Source image with keypoints
        if 'source' in inp:
            source = inp['source'].data.cpu()
            kp_source = out['kp_source']['value'].data.cpu().numpy()
            source = np.transpose(source, [0, 2, 3, 1])
            images.append((source, kp_source))

        if 'compressed_source' in out:
            comp_source = out['compressed_source'].data.cpu().numpy()
            comp_source = np.transpose(comp_source, [0, 2, 3, 1])
            images.append(comp_source)

        # Driving image with keypoints
        if 'driving' in inp:
            kp_driving = out['kp_driving']['value'].data.cpu().numpy()
            driving = inp['driving'].data.cpu().numpy()
            driving = np.transpose(driving, [0, 2, 3, 1])
            images.append(driving)
            images.append((driving, kp_driving))
        
        # hevc base layer
        if 'hevc' in inp:
            hevc = inp['hevc'].data.cpu().numpy()
            hevc = np.transpose(hevc, [0, 2, 3, 1])
            images.append(hevc)



        # Result with and without keypoints
        if 'prediction' in out:
            prediction = out['prediction'].data.cpu().numpy()
            prediction = np.transpose(prediction, [0, 2, 3, 1])
            if 'kp_norm' in out:
                kp_norm = out['kp_norm']['value'].data.cpu().numpy()
                images.append((prediction, kp_norm))
            images.append(prediction)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
