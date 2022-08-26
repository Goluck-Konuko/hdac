from skimage import img_as_ubyte, img_as_float32
import numpy as np
import subprocess
import imageio
import shutil
import torch
import os

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

def compute_bitrate(bits, fps,frames):
    return ((bits*8*fps)/(1000*frames))

class HEVC:
    '''
        HEVC HM CODEC WRAPPER
    '''
    def __init__(self, qp = 50,fps=30,frame_dim=(256,256),gop_size=10, config='models/utils/hevc_hm/config_template.cfg', sequence = None,out_path='hevc_logs/'):
        self.qp = qp
        self.fps = fps
        self.n_frames = gop_size
        self.frame_dim = frame_dim
        self.skip_frames = 0

        self.input = sequence
        self.out_path = out_path
        self.config_name = 'hevc_'+str(qp)+'.cfg'
        self.config_path = config

        #inputs
        self.in_mp4_path = self.out_path+'in_video_'+str(self.qp)+'.mp4'
        self.in_yuv_path = self.out_path+'in_video_'+str(self.qp)+'.yuv'

        #outputs
        self.ostream_path = self.out_path+'out_'+str(self.qp)+'.bin'
        self.dec_yuv_path = self.out_path+'out_'+str(self.qp)+'.yuv'
        self.dec_mp4_path = self.out_path+'out_'+str(self.qp)+'.mp4'
        
        #logging file
        self.log_path =  self.out_path+'out_'+str(self.qp)+'.log'

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        self.config_out_path = self.out_path+self.config_name
        self._create_config()

        #create yuv video
        self._create_mp4()
        self._mp4_2_yuv()

		
    def _create_config(self):
        '''
            Creates a configuration file for HEVC encoder
        '''
        with open(self.config_path, 'r') as file:
            template = file.read()
        #print(template)
        template = template.replace('inputYUV', str(self.in_yuv_path))
        template = template.replace('outStream', str(self.ostream_path))
        template = template.replace('outYUV', str(self.dec_yuv_path))
        template = template.replace('inputW', str(self.frame_dim[0]))
        template = template.replace('inputH', str(self.frame_dim[1]))
        template = template.replace('inputNrFrames', str(self.n_frames))
        template = template.replace('inputSkip', str(self.skip_frames))
        template = template.replace('inputFPS', str(self.fps))
        template = template.replace('setQP', str(self.qp))
        with open(self.config_out_path, 'w+') as cfg_file:
            cfg_file.write(template)


    def _create_mp4(self):
        frames = [img_as_ubyte(frame) for frame in self.input]
        writer = imageio.get_writer(self.in_mp4_path, format='FFMPEG', mode='I',fps=self.fps, codec='libx264',pixelformat='yuv420p', quality=10)
        for frame in frames:
            writer.append_data(frame)
        writer.close()	
		
    def _mp4_2_yuv(self):
        #check for yuv video in target directory
        subprocess.call(['ffmpeg','-nostats','-loglevel','error','-i',self.in_mp4_path,self.in_yuv_path, '-r',str(self.fps)])
		
    def _yuv_2_mp4(self):
        cmd = ['ffmpeg','-nostats','-loglevel','error', '-f', 'rawvideo', '-pix_fmt','yuv420p','-s:v', '256x256', '-r', str(self.fps), '-i', self.dec_yuv_path,  self.dec_mp4_path]
        subprocess.call(cmd)
	
    def _load_sequences(self):
        original = imageio.mimread(self.input_path, memtest=False)
        decoded = imageio.mimread(self.dec_mp4_path, memtest=False)
        return original, decoded        
		
    def _get_hevc_frames(self):
        #convert yuv to mp4
        self._yuv_2_mp4()
        frames = imageio.mimread(self.dec_mp4_path)
        hevc_frames = torch.tensor(np.array([np.array([img_as_float32(frame) for frame in frames]).transpose((3, 0, 1, 2))]), dtype=torch.float32)
        return hevc_frames

		
    def encode(self):
        cmd = ["models/utils/hevc_hm/hm_16_15_regular/bin/TAppEncoderStatic", "-c", self.config_out_path,"-i", self.in_yuv_path]
        with open(self.log_path, 'w+') as out:
            subprocess.call(cmd, stdout=out)
        bytes = os.path.getsize(self.ostream_path)
        bitrate = compute_bitrate(bytes,self.fps,self.n_frames)
        hevc_frames = self._get_hevc_frames()
        shutil.rmtree(self.out_path)
        return hevc_frames, bitrate, bytes*8

				
