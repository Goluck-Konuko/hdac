import torch
import os
import json
import numpy as np
import imageio
from models import DAC, HDAC
from tqdm import trange
from utils.bpg import BPG
from utils.logger import Logger
from torch.utils.data import DataLoader
from skimage import img_as_ubyte, img_as_float32

# from ffcv.loader import Loader
import wandb

def write_data(metrics, metrics_dir):
    with open(metrics_dir,'w') as data:
        json.dump(metrics, data)

def read_data(metrics_dir):
    if os.path.isfile(metrics_dir):
        with open(metrics_dir,'r') as data:
            metrics = json.load(data)
    else:
        metrics = {}

    return metrics
def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_frame(x):
    x = torch.squeeze(x, dim=0).data.cpu().numpy()
    return np.transpose(x, [1, 2, 0])

def compute_bitrate(bits, fps,frames):
    return ((bits*fps)/(1000*frames))


def compress_source_frame(x, qp=30):
    #convert from tensor to uint8 of dimension HxWxC
    bpg = BPG()
    src = get_frame(x)
    #compress
    out = bpg.run(img_as_ubyte(src), qp)
    #convert decoded back to float32 tensor of dimension BxCxHxW
    src_hat = torch.tensor([img_as_float32(out['rec']).transpose(2,0,1)], dtype=torch.float32)
    out['source'] = to_cuda(src_hat)
    return out

def test(config,dataset,generator, kp_detector, checkpoint, log_dir, model_id):
    train_params = config['train_params']  
    
    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint,generator=generator, kp_detector=kp_detector)
    else:
        start_epoch = 0


    if config['dataset_params']['base_layer']:
        dac = HDAC(kp_detector = kp_detector, generator = generator,  config=config)
    else:
        dac = DAC(kp_detector = kp_detector, generator = generator,  config=config) #freeze pretrained generators
    

    if torch.cuda.is_available():
        dac = dac.cuda() #, device_ids=device_ids)
    dac.eval()

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3, drop_last=True)
    # print(len(dataloader))
    
    alpha= True
    seq=0
    perp, h_perp, br = [],[],[]
    metrics_dir = f'{log_dir}/test_metrics.json'
    metrics = read_data(metrics_dir)
    hevc_brs = {'38':8.41, '40':6.62, '42':5.26, '46':4.22, '48':3.51, '50':2.98}

    n_frames = 5
    fps = 20
    qp = 35
    hevc = 50
    idx = 0
    for x in dataloader:
        video = x['video']
        s_name = x['name'][-1].split('.')[0]
        log_dir += f'_{s_name}'
        
        with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
            dt = {'source': x['source'], 'driving':x['driving']}
            if 'hevc' in x:
                dt['hevc'] = x['hevc']
            generated = dac(dt)
            logger.log_epoch(idx, inp=dt, out=generated, save=True)
        idx += 1      
        break
            