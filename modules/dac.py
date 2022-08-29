from operator import ge
import torch
import time
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from skimage import img_as_ubyte, img_as_float32
from .utils.bitstream import KeypointCompress, write_enh_bitstream
from .utils.bpg import BPG, compute_metrics
from .metrics.metrics import Metrics 
from .utils.visualizer import Visualizer
from .utils.hevc import HEVC
from scipy.spatial import ConvexHull

def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def detach_frame(x):
    x = torch.squeeze(x, dim=0).data.cpu().numpy()
    return img_as_ubyte(np.transpose(x, [1, 2, 0]))

def compute_bitrate(bits, fps,frames):
    return ((bits*fps)/(1000*frames))


def psnr(org, dec):
    mse_val = np.mean((org[:,:,0]-dec[:,:,0])**2)		
    p_value = 10*np.log10(255**2/mse_val)
    return p_value

	

class DAC(nn.Module):
    """
    Base Deep Animation Coding:: 
        ::Inputs - Group of Pictures
        ::Outputs - 
    """
    def __init__(self, kp_detector=None, generator=None, config=None, adaptive=True, threshold=30,source_quality=6):
        super(DAC, self).__init__()
        #Pretrained models
        self.kp_detector = kp_detector
        self.generator = generator

        ## Coding tools::
        # Keypoint compression
        self.kp_compressor = KeypointCompress(num_kp=config['model_params']['common_params']['num_kp'])
        # Source compression
        self.bpg = BPG()
        # Metrics computation
        self.monitor = Metrics()
        #output visualization
        self.visualizer = Visualizer(**config['visualizer_params'])

        #coding params
        self.adaptive =adaptive 
        self.threshold = threshold

        self.relative_jacobian = False
        self.relative_movement = True
        self.adapt_movement_scale = True
        self.bitstream = {}

    def compress_source_frame(self,x, qp=30):
        #convert from tensor to uint8 of dimension HxWxC
        src = detach_frame(x)
        #compress
        out = self.bpg.run(img_as_ubyte(src), qp)
        #convert decoded back to float32 tensor of dimension BxCxHxW
        src_hat = torch.tensor(np.array([img_as_float32(out['rec']).transpose(2,0,1)]), dtype=torch.float32)
        out['source'] = to_cuda(src_hat)
        return out

    
    def compress(self,gop, metrics = ['psnr','ssim','ms_ssim'],**kwargs):
        gop_size=kwargs['gop_size']
        source_qp=kwargs['source_qp']
        fps=kwargs['fps']
        out_dir=kwargs['out_dir']
        compute_metrics = kwargs['compute_metrics']
        start = time.time()
        original_video, decoded_video, visualization = [],[],[]
        
        # compress source frame
        source_info = self.compress_source_frame(gop[:,:,0,:,:], qp=source_qp)
        source = to_cuda(source_info['source'])
        src_bits = source_info['bit_size']
        avg_psnr = psnr(detach_frame(gop[:,:,0,:,:]), detach_frame(source))

        original_video.append(detach_frame(gop[:,:,0,:,:]))
        decoded_video.append(detach_frame(source))

        kp_source = self.kp_detector(source)
        print("COMPRESSING GOP...")

        _,_,frames,_,_ = gop.shape
        kp_reference = kp_source
        if not self.adaptive:
            for idx in tqdm(range(1,frames)):
                driving = to_cuda(gop[:,:,idx,:,:])
                #DRIVING KP QUANTIZATION, ENCODING AND RECONSTRUCTION
                kp_target = self.kp_detector(driving)
                kp_driving, _ = self.kp_compressor.encode_kp(kp_reference, kp_target)
                kp_driving['value'] = to_cuda(kp_driving['value'])
                if 'jacobian' in kp_driving:
                    kp_driving['jacobian'] = to_cuda(kp_driving['jacobian'])

                generated = self.generator(source,kp_source= kp_source,kp_driving =kp_driving)
                vis_info = {"source":gop[:,:,0,:,:],
                            'compressed_source': source,
                            "driving":driving,
                            "kp_source":kp_source,
                            "kp_driving": kp_driving,
                            "prediction":generated['prediction'],
                            "flow": generated['flow']}
                viz_img = self.visualizer.visualize(**vis_info)
                
                visualization.append(viz_img)
                decoded_video.append(detach_frame(generated['prediction']))
                original_video.append(detach_frame(gop[:,:,idx,:,:]))
                avg_psnr = (psnr(detach_frame(gop[:,:,idx,:,:]), detach_frame(generated['prediction']))+avg_psnr)/2
                kp_reference = kp_driving
        else:
            for idx in tqdm(range(1,frames)): 
                if idx>1 and avg_psnr < float(self.threshold):
                    #load new source frame if avg psnr is below threshold
                    source_info = self.compress_source_frame(gop[:,:,idx,:,:], qp=source_qp)
                    source = to_cuda(source_info['source'])
                    src_bits += source_info['bit_size']

                    decoded_video.append(detach_frame(source))
                    original_video.append(detach_frame(gop[:,:,idx,:,:]))
                    kp_source = self.kp_detector(source)
                    psnr_val = psnr(detach_frame(gop[:,:,idx,:,:]), detach_frame(source))
                    avg_psnr = (psnr_val+avg_psnr)/2
                else:
                    driving = to_cuda(gop[:,:,idx,:,:])
                    kp_target = self.kp_detector(driving)
                    kp_driving, _ = self.kp_compressor.encode_kp(kp_source, kp_target, out_dir=out_dir)
                    kp_driving['value'] = to_cuda(kp_driving['value'])
                    if 'jacobian' in kp_driving:
                        kp_driving['jacobian'] = to_cuda(kp_driving['jacobian'])

                    generated = self.generator(source,kp_source= kp_source,kp_driving =kp_driving)
                    vis_info = {"source":source,
                                "driving":driving,
                                "kp_source":kp_source,
                                "kp_driving": kp_driving,
                                "prediction":generated['prediction']}
                    viz_img = self.visualizer.visualize(**vis_info)
                
                    visualization.append(viz_img)
                    decoded_video.append(detach_frame(generated['prediction']))
                    original_video.append(detach_frame(gop[:,:,idx,:,:]))
                    avg_psnr = (psnr(detach_frame(gop[:,:,idx,:,:]), detach_frame(generated['prediction']))+avg_psnr)/2
                            
        kp_bits, cdf_info = self.kp_compressor.get_bitstream()
        self.kp_compressor.reset()

        
        encoding_time = np.round(time.time()-start,2)
        out = {'original': original_video, 'decoded':decoded_video,'visualization': visualization}
        if compute_metrics:
            print("COMPUTING METRICS.....")
            metrics = self.monitor.compute_metrics(original_video, decoded_video, metrics=metrics)
            bitrate = compute_bitrate(kp_bits+src_bits, fps,gop_size)
            bit_info ={'bitrate': bitrate,
                        'time': encoding_time,
                        'src_bits': src_bits,
                        'kp_bits': kp_bits}
            out.update({ 'metrics':metrics, 'cdf_info': cdf_info, 'bit_info': bit_info})
        return out
    
    def forward(self, x):
        kp_driving = self.kp_detector(x['driving'])
        kp_source = self.kp_detector(x['source'])
        generated = self.generator(x['source'], kp_source= kp_source,kp_driving =kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        return generated
    

class HDAC(DAC):
    def __init__(self, kp_detector=None, generator=None, config=None):
        super().__init__(kp_detector, generator, config)

    def _get_frames(self, gop, gop_size):
        frames = []
        for idx in range(gop_size):
            frame = detach_frame(gop[:,:,idx,:,:])
            frames.append(frame)
        return frames

    def compress(self,gop, metrics = ['psnr','ssim','ms_ssim','vif','perp'],**kwargs):
        #hevc_gop=None,
        gop_size=kwargs['gop_size']
        source_qp=kwargs['source_qp']
        hevc_qp=kwargs['hevc_qp']
        fps=kwargs['fps']
        compute_metrics = kwargs['compute_metrics']
        start  = time.time()
        ##Create HEVC enhancement layer
        original_video = self._get_frames(gop, gop_size)
        print("Creating HEVC STREAM...")
        hevc_coder = HEVC(qp=hevc_qp,gop_size = gop_size,fps=fps,sequence=original_video)
        hevc_frames, br, hevc_bits = hevc_coder.encode()

        original_video,hevc_video, decoded_video, visualization = [],[],[],[]
        
        # compress source frame
        source_info = self.compress_source_frame(gop[:,:,0,:,:], qp=source_qp) #
        source = to_cuda(source_info['source']) #
        src_bits = source_info['bit_size']

        hevc_video.append(detach_frame(hevc_frames[:,:,0,:,:]))
        original_video.append(detach_frame(gop[:,:,0,:,:]))
        decoded_video.append(detach_frame(source))

        kp_source = self.kp_detector(source)
        print("Compressing with H-DAC...")

        _,_,frames,_,_ = gop.shape

        for idx in tqdm(range(1,frames)):
            driving = to_cuda(gop[:,:,idx,:,:])
            hevc = to_cuda(hevc_frames[:,:,idx,:,:])

            kp_target = self.kp_detector(driving)
            kp_driving, _ = self.kp_compressor.encode_kp(kp_source, kp_target)
            kp_driving['value'] = to_cuda(kp_driving['value'])
            if 'jacobian' in kp_driving:
                kp_driving['jacobian'] = to_cuda(kp_driving['jacobian'])

            generated = self.generator(source,base_layer=hevc,kp_source= kp_source,kp_driving =kp_driving)
            vis_info = {"source":source,
                        "driving":driving,
                        "hevc": hevc,
                        "kp_source":kp_source,
                        "kp_driving": kp_driving,
                        "prediction":generated['prediction']}
            viz_img = self.visualizer.visualize(**vis_info)
            
            visualization.append(viz_img)
            decoded_video.append(detach_frame(generated['prediction']))
            original_video.append(detach_frame(gop[:,:,idx,:,:]))
            hevc_video.append(detach_frame(hevc_frames[:,:,idx,:,:]))
        
        kp_bits, cdf_info = self.kp_compressor.get_bitstream()
        self.kp_compressor.reset()
        total_bits = kp_bits +hevc_bits + src_bits

        
        encoding_time = np.round(time.time()-start,2)
        out = {'original': original_video, 'decoded':decoded_video,'visualization': visualization}
        if compute_metrics:
            print("COMPUTING METRICS.....")
            metrics = self.monitor.compute_metrics(original_video[1:], decoded_video[1:], metrics=metrics)
            hevc_metrics = self.monitor.compute_metrics(original_video[1:], hevc_video[1:], metrics=metrics)
            bitrate = compute_bitrate(total_bits, fps,gop_size)

            bit_info ={'bitrate': bitrate,
                        'time': encoding_time,
                        'src_bits': src_bits,
                        'kp_bits': kp_bits,
                        'hevc_bits': hevc_bits}

            out.update({'metrics':metrics,'hevc_metrics':hevc_metrics, 'cdf_info': cdf_info,'bit_info': bit_info})
        return out

    def forward(self, x):
        kp_driving = self.kp_detector(x['driving'])
        kp_source = self.kp_detector(x['source'])
        generated = self.generator(x['source'], base_layer=x['hevc'],kp_source= kp_source,kp_driving =kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        return generated



    