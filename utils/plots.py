import numpy as np
import matplotlib.pyplot as plt

class Plots:
	def __init__(self,profile, thresh, qp,out_path,version=10, dac=None, hevc=None,metrics=['psnr', 'ssim','ms_ssim','vif']):
		self.dac_metrics = dac
		self.hevc_metrics = hevc
		self.seq_len = min(len(self.dac_metrics['psnr']), len(self.hevc_metrics["psnr"]))
		#print(len(self.dac_metrics['psnr']), len(self.hevc_metrics['psnr']))
		self.metrics = metrics
		self.profile = profile
		self.thresh = thresh
		self.qp = qp
		self.version=version
		self.out_path = out_path
		
	def plot(self, metric):
		if self.dac_metrics is not None:
			dac_metric = self.dac_metrics[metric]
			if metric =='psnr':
				plt.plot(range(self.seq_len), dac_metric[:self.seq_len], label = 'DAC || {}: {} dB || BR: {} kbps || Var: {}'.format(metric.upper(), round(np.mean(np.array(dac_metric)),2),self.dac_metrics['bitrate'],round(np.var(dac_metric, axis=0),3)))
			else:
				plt.plot(range(self.seq_len), dac_metric[:self.seq_len], label = 'DAC || {}: {} || BR: {} kbps || Var: {}'.format(metric.upper(), round(np.mean(np.array(dac_metric)),2),self.dac_metrics['bitrate'],round(np.var(dac_metric, axis=0),3)))
		if self.hevc_metrics is not None:
			hevc_metric = self.hevc_metrics[metric]
			if metric == 'psnr':
				plt.plot(range(self.seq_len), hevc_metric[:self.seq_len], label='HEVC || {}: {} dB || BR: {} kbps || Var: {}'.format(metric.upper(), round(np.mean(np.array(hevc_metric)),2),self.hevc_metrics['bitrate'], round(np.var(hevc_metric, axis=0),3)))
			else:
				plt.plot(range(self.seq_len), hevc_metric[:self.seq_len], label='HEVC || {}: {} || BR: {} kbps || Var: {}'.format(metric.upper(), round(np.mean(np.array(hevc_metric)),2),self.hevc_metrics['bitrate'], round(np.var(hevc_metric, axis=0),3)))
		plt.title('{} Temporal Variation'.format(metric.upper()))
		plt.legend()
		plt.xlabel("Frames")
		plt.savefig(self.out_path +'/{}_{}_{}_{}_{}.png'.format(metric,self.profile,self.thresh,self.qp, self.version))
		plt.close()
			
	def generate(self, target='all'):
		if target == 'all':
			for metric in self.metrics:
				self.plot(metric)
		else:
			if target.lower() in self.metrics:
				self.plot(target)
			else:
				print("{} or {} not found in metrics list!".format(target.upper(), target.lower()))
				
		

'''
def plot_curve(dac, hevc, metric='PSNR'):
	plt.plot(range(len(dac[0]))


def _plot_curves(self, bitrate, hevc_bitrate):
		plt.plot(range(len(self.seq_psnr)),self.seq_psnr)
		plt.plot(range(len(self.seq_hevc_psnr)),self.seq_hevc_psnr)
		plt.title("|--PSNR Temporal Variation--|")
		plt.xlabel("Frames")
		plt.ylabel("PSNR(dB)")
		plt.legend(('DAC || PSNR: {} dB || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_psnr)),2),bitrate, round(np.var(self.seq_psnr, axis=0),3)),'HEVC || PSNR: {} dB || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_hevc_psnr)),2),hevc_bitrate, round(np.var(self.seq_hevc_psnr, axis=0),3))))
		plt.savefig(self.result_metrics_path +'/psnr_'+str(self.profile)+'_'+str(self.thresh)+'_'+str(self.qp)+'.png')
		plt.close()	
		
		plt.plot(range(len(self.seq_ssim)),self.seq_ssim)
		plt.plot(range(len(self.seq_hevc_ssim)),self.seq_hevc_ssim)
		plt.title("|--SSIM Temporal Variation--|")
		plt.xlabel("Frames")
		plt.ylabel("SSIM")
		plt.legend(('DAC || SSIM: {}  || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_ssim)),2),bitrate, round(np.var(self.seq_ssim, axis=0),3)),'HEVC || SSIM: {} || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_hevc_ssim)),2),hevc_bitrate, round(np.var(self.seq_hevc_ssim, axis=0),3))))
		plt.savefig(self.result_metrics_path +'/ssim_'+str(self.profile)+'_'+str(self.thresh)+'_'+str(self.qp)+'.png')
		plt.close()	
		
		plt.plot(range(len(self.ms_ssim)),self.ms_ssim)
		plt.plot(range(len(self.hevc_ms_ssim)),self.hevc_ms_ssim)
		plt.title("|--MS-SSIM Temporal Variation--|")
		plt.xlabel("Frames")
		plt.ylabel("MS-SSIM")
		plt.legend(('DAC || MS-SSIM: {} || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.ms_ssim)),2),bitrate, round(np.var(self.seq_ssim),3)),'HEVC || MS-SSIM: {} || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.hevc_ms_ssim)),2),hevc_bitrate, round(np.var(self.hevc_ms_ssim, axis=0),3))))
		plt.savefig(self.result_metrics_path +'/ms_ssim_'+str(self.profile)+'_'+str(self.thresh)+'_'+str(self.qp)+'.png')
		plt.close()	
		
		plt.plot(range(len(self.seq_perp_loss)),self.seq_perp_loss)
		plt.plot(range(len(self.seq_hevc_perp_loss)),self.seq_hevc_perp_loss)
		plt.title("|--Perceptual Loss Temporal Variation--|")
		plt.xlabel("Frames")
		plt.ylabel("Perceptual Loss")
		plt.legend(('DAC || P. Loss: {}  || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_perp_loss)),2),bitrate, round(np.var(self.seq_perp_loss, axis=0),3)),'HEVC || P.loss: {} || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_hevc_perp_loss)),2),hevc_bitrate, round(np.var(self.seq_hevc_perp_loss, axis=0),3))))
		plt.savefig(self.result_metrics_path + "/perp_"+str(self.profile)+'_'+str(self.thresh)+'_'+str(self.qp)+".png")
		plt.close()
		
		plt.plot(range(len(self.seq_vif)),self.seq_vif)
		plt.plot(range(len(self.seq_hevc_vif)),self.seq_hevc_vif)
		plt.title("|--Visual Information Fidelity--|")
		plt.xlabel("Frames")
		plt.ylabel("V.I.F")
		plt.legend(('DAC || VIF: {} dB || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_vif)),2),bitrate, round(np.var(self.seq_vif, axis=0),3)),'HEVC || VIF: {} || BR: {} kbps || Var: {}'.format(round(np.mean(np.array(self.seq_hevc_vif)),2),hevc_bitrate, round(np.var(self.seq_hevc_vif, axis=0),3))))
		plt.savefig(self.result_metrics_path + "/vif_"+str(self.profile)+'_'+str(self.thresh)+'_'+str(self.qp)+".png")
		plt.close()
'''
