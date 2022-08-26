
import matplotlib
matplotlib.use('Agg')

import torch
import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from models import GeneratorDAC,GeneratorHDAC, KPD, MultiScaleDiscriminator
from utils.dataset import FramesDataset
from train import train
from test import test
from compression import compress

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning, module='torch')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train","compress", "test"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    #coding params
    parser.add_argument("--start_frame", default=0, type=int, help="Starting frame in the encoded video sequence")
    parser.add_argument("--end_frame", default=32, type=int, help="Last frame in the encoded video sequence")
    parser.add_argument("--source_qp", default=30, type=int, help="Compression QP for the reference frame")
    parser.add_argument("--hevc_qp", default=50, type=int,help="Compression level for the enhancement layer")

    

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_id = os.path.basename(opt.config).split('.')[0]
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += '_' + strftime("%d_%m_%y_%H.%M", gmtime())
    
    #import Generator module
    if model_id == "hdac":
        generator = GeneratorHDAC(**config['model_params']['common_params'],**config['model_params']['generator_params'])
    else:
        generator = GeneratorDAC(**config['model_params']['common_params'],**config['model_params']['generator_params'])
    
    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])

    #import keypoint detector
    kp_detector = KPD(**config['model_params']['common_params'],**config['model_params']['kp_detector_params'], training= opt.mode == 'train')
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
    
    #import discriminator
    if opt.mode =="train":
        discriminator = MultiScaleDiscriminator(**config['model_params']['common_params'],**config['model_params']['discriminator_params'])
        if torch.cuda.is_available():
            discriminator.to(opt.device_ids[0])
        
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    if opt.mode == 'train':
        print("Training...")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)
            copy('models/components/generator.py', log_dir)
        #pass config, generator, kp_detector and discriminator to the training module
        train(config,dataset, generator,kp_detector,discriminator, opt.checkpoint, log_dir)
    # elif opt.mode == 'test':
    #     print("Testing..")
    #     log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
    #     log_dir += '_' + strftime("%d_%m_%y_%H.%M", gmtime())
    #     # if not os.path.exists(log_dir):
    #     #     os.makedirs(log_dir)
    #     test(config, dataset, generator, kp_detector, opt.checkpoint,log_dir,model_id)
    # elif opt.mode == 'compress':
    #     print("Compression mode..")
    #     log_dir = f'compression_logs_qp/{model_id}'
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)
        
    #     coding_params = {
    #                     'frames': opt.frames,
    #                     'gop_size': opt.gop_size,
    #                     'source_qp': opt.source_qp,
    #                     'hevc_qp' :  opt.hevc_qp,
    #                     'metrics': ['psnr','ms_ssim','perp'] #['ms_ssim','vif','perp']
    #                     }
    #     compress(config, dataset, generator, kp_detector, opt.checkpoint,log_dir,model_id, **coding_params)

    
