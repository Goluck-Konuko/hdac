from skimage.metrics import structural_similarity
# from modules import Vgg19, ImagePyramide
# from modules.util.generic_blocks import AntiAliasInterpolation2d

from skimage import  img_as_float
import scipy.signal
import scipy.ndimage
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from skvideo.measure import msssim
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
# import lpips


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale),mode='bilinear', align_corners=True)

        return out
        
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ComputeAverage:
	s_psnr = AverageMeter()
	s_perp = AverageMeter()
	s_sim = AverageMeter()
	s_ms_ssim = AverageMeter()
	s_vif = AverageMeter()

class Metrics:
	def __init__(self, original=None, decoded = None, seq=True, gop_size=10):
		self.original = original
		self.decoded = decoded
		self.seq = seq
		self.psnr=0
		self.wpsnr=0
		self.ssim=0
		self.ms_ssim=0
		self.vif=0
		self.vmaf=0
		self.perp_loss=0
		self.pim_loss=0
		self.nlp_loss=0
		self.lpips_loss  = 0
		self.s_psnr, self.s_wpsnr, self.s_ssim, self.s_ms_ssim, self.s_vif, self.s_vmaf, self.s_perp_loss, self.s_pim_loss, self.s_nlp_loss, self.s_lpips_loss  = [],[],[],[],[],[],[],[],[],[]
		self.error_video = []
		self.error_img = None
		self.gop_size = gop_size
		self.avg_monitor = ComputeAverage()
		
	def _presets(self,org=None, dec=None, seq=None):
		if org is not None:
			self.original = org
		if dec is not None:
			self.decoded = dec
		if seq is not None:
			self.seq = seq
		
	def _mse(self, org, dec):
		return np.mean((org-dec)**2)
		
	def _mscale(self,ref, dist):
		sigma_nsq=4
		eps = 1e-10
		num = 0.0
		den = 0.0
		for scale in range(1, 5):
			N = 2**(4-scale+1) + 1
			sd = N/5.0
			if (scale > 1):
				ref = scipy.ndimage.gaussian_filter(ref, sd)
				dist = scipy.ndimage.gaussian_filter(dist, sd)
				ref = ref[::2, ::2]
				dist = dist[::2, ::2]        
			mu1 = scipy.ndimage.gaussian_filter(ref, sd)
			mu2 = scipy.ndimage.gaussian_filter(dist, sd)
			mu1_sq = mu1 * mu1
			mu2_sq = mu2 * mu2
			mu1_mu2 = mu1 * mu2
			sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
			sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
			sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
			sigma1_sq[sigma1_sq<0] = 0
			sigma2_sq[sigma2_sq<0] = 0
        	
			g = sigma12 / (sigma1_sq + eps)
			sv_sq = sigma2_sq - g * sigma12
			g[sigma1_sq<eps] = 0
			sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
			sigma1_sq[sigma1_sq<eps] = 0
        
			g[sigma2_sq<eps] = 0
			sv_sq[sigma2_sq<eps] = 0
			sv_sq[g<0] = sigma2_sq[g<0]
			g[g<0] = 0
			sv_sq[sv_sq<=eps] = eps
        	
			num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
			den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
		vifp = num/den
		if np.isnan(vifp):
			return 1.0
		else:
			return vifp
	
	def _psnr(self, org=None, dec=None, seq=None):	
		print("Computing PSNR..")	
		if self.seq:
			ref = [frame[:,:,0] for frame in self.original]
			dec = [frame[:,:,0] for frame in self.decoded]
			for idx in range(len(dec)):			
				mse_val = self._mse(ref[idx], dec[idx])
				psnr_val = 10*np.log10(255**2/mse_val)
				self.s_psnr.append(psnr_val)
			return self.s_psnr		
		else:
			mse_val = self._mse(self.original[:,:,0], self.decoded[:,:,0])
			self.psnr = 10*np.log10(255**2/mse_val)
			return self.psnr
		
	def _ssim(self, org=None, dec=None, seq=None):
		'''structual similarity index'''
		print("Computing SSIM..")	
		if self.seq:
			ref = [frame[:,:,0] for frame in self.original]
			dec = [frame[:,:,0] for frame in self.decoded]
			for idx in range(len(self.decoded)):
				ssim_val = structural_similarity(ref[idx], dec[idx], data_range = np.amax(dec)-np.amin(dec), size_average=False)
				self.s_ssim.append(ssim_val)
			return self.s_ssim
		else:
			self.ssim = structural_similarity(self.original[:,:,0], self.decoded[:,:,0], data_range = np.amax(dec)-np.amin(dec), size_average=False)
			return self.ssim
	
	def _ms_ssim(self, org=None, dec=None, seq=None):
		'''Multiscale structual similarity index'''
		print("Computing MS-SSIM..")	
		if self.seq:
			frames = len(self.decoded)
			ref = [frame[:,:,0] for frame in self.original[0:frames]]
			dec = [frame[:,:,0] for frame in self.decoded] 
			seq_ssim = msssim(ref,dec)
			self.s_ms_ssim = [x.astype(float) for x in seq_ssim]
			return self.s_ms_ssim
		else:
			self.ms_ssim = msssim(self.original[:,:,0],self.decoded[:,:,0]).astype(float)
			return self.ms_ssim
			
	def _vif(self,org=None, dec=None, seq=None):
		'''Multi-scale Visual information fidelity'''
		print("Computing VIF..")	    
		if self.seq:
			for idx in tqdm(range(len(self.decoded))):
				vif = self._mscale(self.original[idx],self.decoded[idx])
				self.s_vif.append(vif)
			return self.s_vif
		else:
			self.vif = self._mscale(self.original, self.decoded)
			return self.vif
			
	# def _lpips(self,org=None, dec=None,seq = None):
	# 	self._presets(org, dec,seq)
	# 	loss_net = lpips.LPIPS(net='vgg')
	# 	if torch.cuda.is_available():
	# 		loss_net = loss_net.cuda()
	# 	if self.seq:
	# 		for idx in tqdm(range(len(self.decoded))):		
	# 			org_frame = img_as_float(self.original[idx])
	# 			dec_frame = img_as_float(self.decoded[idx])
	# 			org = torch.tensor(org_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
	# 			dec = torch.tensor(dec_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
				
	# 			if torch.cuda.is_available():
	# 				org = org.cuda()
	# 				dec = dec.cuda()
				
	# 			loss = loss_net(org, dec)
	# 			self.s_lpips_loss.append(loss.item())
	# 		return self.s_lpips_loss
	# 	else:
	# 		org = torch.tensor(self.original[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
	# 		dec = torch.tensor(self.decoded[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
	# 		if torch.cuda.is_available():
	# 			org = org.cuda()
	# 			dec = dec.cuda()			
	# 		self.lpips_loss = loss_net(org, dec).item() 
	# 		return self.lpips_loss

		
	def _perceptual_loss(self,org=None, dec=None,seq = None):
		loss_weights = [10, 10, 10, 10, 10]
		scales  = [1, 0.5, 0.25,0.125]
		vgg = Vgg19()
		pyramid = ImagePyramide(scales, 3)
		if torch.cuda.is_available():
			vgg = vgg.cuda()
			pyramid = pyramid.cuda()
		if self.seq:
			for idx in tqdm(range(len(self.decoded))):		
				org_frame = img_as_float(self.original[idx])
				dec_frame = img_as_float(self.decoded[idx])
				org = torch.tensor(org_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
				dec = torch.tensor(dec_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
				
				if torch.cuda.is_available():
					org = org.cuda()
					dec = dec.cuda()
				
				pyramide_real = pyramid(org)
				pyramide_generated = pyramid(dec)
				total_loss = 0
				for scale in scales:
					x_vgg = vgg(pyramide_generated['prediction_' + str(scale)])
					y_vgg = vgg(pyramide_real['prediction_' + str(scale)])
					for i, weight in enumerate(loss_weights):
						value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
						if not torch.isnan(value):
							total_loss += loss_weights[i] * value
				self.s_perp_loss.append(total_loss.item())
			return self.s_perp_loss
		else:
			org = torch.tensor(self.original[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
			dec = torch.tensor(self.decoded[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
			if torch.cuda.is_available():
				org = org.cuda()
				dec = dec.cuda()
			
			pyramide_real = pyramid(org)
			pyramide_generated = pyramid(dec)
			total_loss = 0

			x_vgg = vgg(pyramide_generated['prediction_' + str(1)])
			y_vgg = vgg(pyramide_real['prediction_' + str(1)])
			total_loss += torch.abs(x_vgg[3] - y_vgg[3].detach()).mean()
			self.perp_loss =  total_loss.item()
			return self.perp_loss
		

		
	def compute_sequence_metrics(self, org=None, dec=None, seq=True,metrics=['psnr','ssim','ms_ssim','vif','perp']):
		self._presets(org, dec,seq)
		self.s_psnr,self.error_video, self.s_ssim, self.s_ms_ssim, self.s_vif, self.s_vmaf, self.s_perp_loss, self.s_pim_loss = [],[],[],[],[],[],[],[]
		out = {}
		if 'psnr' in metrics:
			self._psnr()
			out['psnr'] = self.s_psnr #np.round(np.mean(self.s_psnr),3)
		if 'ssim' in metrics:
			self._ssim()
			out['ssim'] = np.round(np.mean(self.s_ssim),3)
		if 'ms_ssim' in metrics:
			self._ms_ssim()
			out['ms_ssim'] = self.s_ms_ssim#np.round(np.mean(self.s_ms_ssim),3)
		if 'vif' in metrics:
			self._vif()
			out['vif'] = self.s_vif
		if 'perp' in metrics:
			self._perceptual_loss()
			out['perp'] = self.s_perp_loss
		return out
	
	def compute_frame_metrics(self, org=None, dec=None, seq=False,metrics=['psnr','ssim','ms_ssim','vif','perp']):
		self._presets(org, dec,seq)
		out = {}
		if 'psnr' in metrics:
			out['psnr'] = self._psnr()
		if 'ssim' in metrics:
			out['ssim'] = self._ssim()
		if 'ms_ssim' in metrics:
			out['ms_ssim'] = self._ms_ssim()
		if 'vif' in metrics:
			out['vif'] = self._vif() 
		if 'perp' in metrics:
			out['perp'] = self._perceptual_loss()
		return out
				

		

