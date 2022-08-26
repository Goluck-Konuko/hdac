from models.dac import compute_bitrate
import torch
import os
import json
import imageio
from models import DAC, HDAC
from torch.utils.data import DataLoader

def get_gops(video, frames ,gop_size):
    _,_,n_frames,_,_ = video.shape
    gops = []
    if gop_size > n_frames:
        raise Exception("Insufficient video length, put longer video or reduce gop size")

    if frames > n_frames:
        print(f"Target: {frames} longer than video sequence: {n_frames} frames: Will return valid {n_frames // gop_size} frames")
        frames = (n_frames//gop_size)*gop_size
        video = video[:,:,:frames,:,:]
    else:
        video = video[:,:,:frames,:,:]

    g_end = gop_size
    for g_start in range(0, n_frames, gop_size):
        if g_end <= n_frames and g_end <= frames:
            gops.append(video[:,:,g_start:g_end,:,:])
            g_end += gop_size
        elif g_start < frames and g_end> frames:
            gops.append(video[:,:,g_start:frames,:,:])
            break
        else:
            break
    return gops
    

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

def compute_bitrate(bits, fps,frames):
    return ((bits*fps)/(1000*frames))

def compress(config,dataset,generator, kp_detector, checkpoint, log_dir, model_id, **kwargs):
    frames=kwargs['frames']
    gop_size=kwargs['gop_size']
    source_qp=kwargs['source_qp']
    hevc_qp = kwargs['hevc_qp']
    target_metrics = kwargs['metrics']

    train_params = config['train_params']  
    
    if checkpoint is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint, map_location=device)
        
        #load generator
        generator.load_state_dict(checkpoint['generator'])

        #load kp detector
        kp_detector.load_state_dict(checkpoint['kp_detector'])

    if config['dataset_params']['base_layer']:
        dac = HDAC(kp_detector = kp_detector, generator = generator,  config=config)
    else:
        dac_params = {
            'adaptive': True,
            'threshold': 33
        }
        dac = DAC(kp_detector = kp_detector, generator = generator,  config=config, **dac_params) #freeze pretrained generators
    

    if torch.cuda.is_available():
        dac = dac.cuda()
    dac.eval()

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3, drop_last=True)

    metrics_dir = f'{log_dir}/test_metrics_hevc.json'
    metrics = read_data(metrics_dir)

    for x in dataloader:
        video = x['video']
        fps = x['fps'].data.cpu().numpy()[-1]
        s_name = x['name'][-1].split('.')[0]
        tgt_dir =log_dir + f'/{s_name}'

        decoded_info = []
        coding_params = {
                    "gop_size":gop_size,
                    "source_qp":source_qp, 
                    "fps":fps,
                    "hevc_qp": hevc_qp 
                    }
        #metrics log name
        m_log_name = f"{coding_params['gop_size']}_{coding_params['source_qp']}"
        gop = video[:,:,:gop_size,:,:]

        if config['dataset_params']['base_layer']:
            m_log_name  += f"_{coding_params['hevc_qp']}"

        out = dac.compress(gop, metrics=target_metrics, **coding_params)

        if s_name in metrics:
            metrics[s_name][m_log_name] = out['metrics']
        else:
            metrics[s_name] = {m_log_name: out['metrics']}

        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)

        if config['dataset_params']['base_layer']:
            imageio.mimsave(tgt_dir+f'/vis_video_{gop_size}_{source_qp}_{hevc_qp}.mp4', out['visualization'], fps=fps)
            imageio.mimsave(tgt_dir+f'/rec_video_{gop_size}_{source_qp}_{hevc_qp}.mp4', out['decoded'], fps=fps)
        else:
            imageio.mimsave(tgt_dir+f'/vis_video_{gop_size}_{source_qp}.mp4', out['visualization'], fps=fps)
            imageio.mimsave(tgt_dir+f'/rec_video_{gop_size}_{source_qp}.mp4', out['decoded'], fps=fps)
        # break
    write_data(metrics, metrics_dir)
            