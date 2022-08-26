import matplotlib.pyplot as plt
import os, sys
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import skimage.io as io

from utils.metrics import *
from utils.curves import *
	
    
def analyser(driving_video, predictions,visualizations, hevc_metrics,total_bytes, fps,driving_video_path,result_video_path, filename,srcs,src_idx):
    #Perform analysis
    frames = len(predictions)
    seq_bitrate = bitrate(total_bytes, fps)
    hevc_metrics[filename]["dac_bitrate_15"] = seq_bitrate         
            
    hevc_psnr = hevc_metrics[filename]["hevc_psnr"]
    hevc_bitrate = hevc_metrics[filename]["hevc_bitrates"]
            
    qps = [15,20,25,30,35,40,45,50]
    qp = find_nearest_hevc(hevc_bitrate, seq_bitrate)
    idx = qps.index(qp)
    hevc_br = hevc_bitrate[idx]
    #find hevc video with closest bitrate
    hevc_video_name = filename +"_"+str(qp)+".mp4"
            
    #load hevc sequence
    hevc_video = imageio.mimread(driving_video_path+"/hevc/"+filename+'/'+ hevc_video_name, memtest=False) 
    hevc_video = [img_as_float(frame) for frame in hevc_video]
            
            
    #psnr metrics
    seq_psnr, error_video = psnr(driving_video, predictions, seq = True)
    seq_hevc_psnr, error_video_hevc = psnr(driving_video, hevc_video, seq = True)
    plot_psnr(seq_psnr,frames,result_video_path, filename, seq_hevc_psnr)
            
    hevc_metrics[filename]["seq_psnr_"+str(srcs)] = seq_psnr
    hevc_metrics[filename]["seq_psnr_hevc_"+str(srcs)] = seq_hevc_psnr
            
    #Perceptual loss
    seq_loss_dac = perceptual_loss(driving_video, predictions, seq = True)
    seq_loss_hevc = perceptual_loss(driving_video, hevc_video, seq = True)
    plot_perp_loss(seq_loss_dac ,frames,result_video_path, filename, seq_loss_hevc )
    mean_perp_hevc = np.mean(np.array(seq_loss_hevc))
    mean_perp_dac = np.mean(np.array(seq_loss_dac))
            
    hevc_metrics[filename]["perp_loss_dac_"+ str(srcs)] = seq_loss_dac
    hevc_metrics[filename]["perp_loss_hevc"+ str(srcs)] = seq_loss_hevc
            
    #ssim metrics
    #seq_ssim = ssim(driving_video, predictions, seq = True)
    #seq_hevc_ssim = ssim(driving_video, hevc_video, seq = True)
    #plot_ssim(seq_ssim,frames,result_video_path,  filename, seq_hevc_ssim)
            
            
    #hevc_metrics[filename]["seq_ssim_dac"] = seq_ssim
    #hevc_metrics[filename]["seq_ssim_hevc"] = seq_hevc_ssim
            
    #plot aggregate rd curve
    mean_psnr_dac = np.mean(np.array(seq_psnr))
    mean_psnr_hevc = np.mean(np.array(seq_hevc_psnr))
    plot_rd(hevc_bitrate, hevc_psnr,seq_bitrate,mean_psnr_dac,hevc_br,mean_psnr_hevc,frames,result_video_path, filename)
            
           
    #compose output video
    output_video = []
    sources = []
    
    if len(sources)==0:
    	sources = driving_video[idx]
    else:
        if len(src_idx)>1:
            for idx in src_idx[-4:]:
    	        sources = np.concatenate((sources, driving_video[idx]), axis=1)
    for idx in range(len(visualizations)):           			
        composite_frame = np.concatenate((img_as_ubyte(sources), visualizations[idx],img_as_ubyte(predictions[idx]),img_as_ubyte(hevc_video[idx])), axis=1)
        output_video.append(composite_frame)
    imageio.mimsave(result_video_path +filename+ "/videos/"+filename+"_vis.mp4", [img_as_ubyte(frame) for frame in output_video], fps=fps)     

    return hevc_br, seq_bitrate,mean_psnr_hevc, mean_psnr_dac,mean_perp_hevc, mean_perp_dac,frames
