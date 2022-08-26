import os
from turtle import shape

from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import pandas as pd
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, FloatField,RGBImageField


def read_video(name, frame_shape):
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

def resize_video_array(video_array, width=64, height=64):
	out = []
	for clip in video_array:
		clip = Image.fromarray(clip)
		clip = clip.resize((width,height), Image.ANTIALIAS)
		out.append(np.asarray(clip))
	return np.array(out)

class FramesDataset:
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir='../datasets', frame_shape=(256, 256, 3),id_sampling=False, is_train=True,
                 random_seed=0,base_layer=True, pairs_list=None, augmentation_params=None, num_sources=2):
        print("working")
        if not is_train:
            self.root_dir = 'coding_tools/inference_videos'
        else:
            self.root_dir = root_dir
        self.videos = os.listdir(self.root_dir)
        self.base_layer = base_layer
        # print(self.videos, self.base_layer)
        # if 'train_hevc' not in self.videos:
        #     print("HEVC base layer data not found..Training without!")
        #     self.base_layer = False
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.num_sources = num_sources

        if os.path.exists(os.path.join(root_dir, 'train')):
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
                #print(len(train_videos))
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(self.root_dir, 'test'))
            if self.base_layer:
                train_hevc_videos = os.listdir(os.path.join(root_dir, 'train_hevc'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
            if self.base_layer:
                self.hevc_base_layer = train_hevc_videos
        else:
            self.videos = test_videos

        self.is_train = is_train
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name)))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
        # print(path)
        video_name = os.path.basename(path)
        if self.base_layer:
            if self.is_train:
                hevc_path = "../datasets/train_hevc/"+name
            else:
                hevc_path = '../datasets/inference_videos/hevc_base/'+ name.split('.')[0]+'.mp4'
            

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=False, size=self.num_sources))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
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
        if self.is_train:
            source = np.array(img_as_float32(video_array[0]), dtype='float32')
            driving = np.array(img_as_float32(video_array[1]), dtype='float32')
            out['source'] = source.transpose((2, 0, 1))
            out['driving'] = driving.transpose((2, 0, 1))
            if self.base_layer:
                hevc_base = np.array(img_as_float32(video_array[-1]), dtype="float32") #assume only the driving frame has a base layer
                out['hevc'] = hevc_base.transpose((2,0,1))
        else:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['source'] = source.transpose((2, 0, 1))
            out['driving'] = driving.transpose((2, 0, 1))
            if self.base_layer:
                hevc_base = np.array(video_array[-1], dtype="float32") #assume only the driving frame has a base layer
                out['hevc'] = hevc_base.transpose((2,0,1))

            video = np.array(video, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))
            # print('vid_length: ', len(video))
            if self.base_layer:
                hevc_video = np.array(hevc_video, dtype='float32')
                out['hevc_video'] = hevc_video.transpose((3, 0, 1, 2))
                # print('hevc_vid_len: ',len(hevc_video))
        # out['name'] = video_name
        # return out
        return (out['source'],out['driving'],out['hevc'])


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]



if __name__ == "__main__":
    H,W,C = 256,256,3
    num_repeats = 1
    base_layer=True
    aug_params = {'flip_param':{'horizontal_flip': True,'time_flip': True},
                            'jitter_param':{'brightness': 0.1,
                                            'contrast': 0.1,
                                            'saturation': 0.1,
                                            'hue': 0.1}}
    dataset = FramesDataset(root_dir='../datasets',base_layer=base_layer, augmentation_params=aug_params)
    dataset = DatasetRepeater(dataset, num_repeats=num_repeats)
    # writer = DatasetWriter('vox.beton',{'source': RGBImageField(),
    #                                     'driving':RGBImageField()},num_workers=4)

    writer =DatasetWriter(f'../datasets/vox_hevc_{num_repeats}.beton',{'source': NDArrayField(shape=(C,H,W), dtype=np.dtype('float32')),
                                        'driving': NDArrayField(shape=(C,H,W), dtype=np.dtype('float32')),
                                        'hevc': NDArrayField(shape=(C,H,W), dtype=np.dtype('float32'))}, num_workers=4)
    # # writer =DatasetWriter('../../datasets/vox.beton',{'source': NDArrayField(shape=(C,H,W), dtype=np.dtype('float32')),
    # #                                     'driving': NDArrayField(shape=(C,H,W), dtype=np.dtype('float32'))}, num_workers=4)

    writer.from_indexed_dataset(dataset)