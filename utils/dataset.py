import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread
import numpy as np
from torch.utils.data import Dataset
from utils.augmentation import AllAugmentationTransform
from typing import Tuple, Dict, List


def read_video(name: str, frame_shape: Tuple[int, int, int]) -> np.array:
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir: str, frame_shape: Tuple[int, int, int]=(256, 256, 3), 
                        is_train: bool=True, base_layer: bool=False,
                        augmentation_params: Dict[Dict[str,bool],Dict[str,bool]]=None, 
                        num_sources: int=2):
        print("working")
        self.root_dir: str = root_dir
        self.videos: List[str] = os.listdir(self.root_dir)
        self.base_layer: bool = base_layer
        self.frame_shape: Tuple[int, int, int] = frame_shape
        self.num_sources: int = num_sources
        self.is_train: bool = is_train
        self.root_dir: str = os.path.join(root_dir, 'train' if is_train else 'test')

        if is_train:
            self.train_videos = os.listdir(os.path.join(root_dir, 'train'))
            self.train_hevc_videos = os.listdir(os.path.join(root_dir, 'train_hevc'))
            if self.base_layer:
                self.root_dir_hevc = os.path.join(root_dir, 'train_hevc')
        else:
            self.test_videos = os.listdir(os.path.join(root_dir, 'test'))

        
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        if self.is_train:
            name = self.train_videos[idx]
            path = os.path.join(self.root_dir, name)
            video_name = os.path.basename(path)
            if self.base_layer:
                hevc_path = os.path.join(self.root_dir_hevc, name) #_{np.random.choice([40,42,44,46,48,50])}

            video = read_video(path, frame_shape=self.frame_shape)
            if self.base_layer:
                hevc_video = read_video(hevc_path, frame_shape = self.frame_shape)
                num_frames = len(hevc_video)
            else:
                num_frames = len(video)
            frame_idx = np.sort(np.random.choice(num_frames, replace=False, size=self.num_sources)) #if self.is_train #else range(num_frames)
            video_array = video[frame_idx]
            
        
            if self.base_layer:
                hevc_video_array = hevc_video[frame_idx]
                video_array = np.array([video_array[0], video_array[1], hevc_video_array[0], hevc_video_array[1]])
            if self.transform is not None:
                video_array = self.transform(video_array)
            
            out = {}
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['source'] = source.transpose((2, 0, 1))
            out['driving'] = driving.transpose((2, 0, 1))
            if self.base_layer:
                hevc_base = np.array(video_array[-1], dtype="float32") #assume only the driving frame has a base layer
                out['hevc'] = hevc_base.transpose((2,0,1))
        return out

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset: FramesDataset, num_repeats: int=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx: int):
        return self.dataset[idx % self.dataset.__len__()]

