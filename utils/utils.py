from modules.generator_2 import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
from collections import namedtuple
from skimage import img_as_ubyte, img_as_float, img_as_int
from utils.bpg import BPGEncoder, BPGDecoder
from utils.float2binary import *
from decimal import *
import contextlib, sys
from utils.arithmetic_coding import ArithmeticEncoder,FlatFrequencyTable,SimpleFrequencyTable,ArithmeticEncoder,ArithmeticDecoder,BitOutputStream, NeuralFrequencyTable
from utils.pyae import ArithmeticEncoding #, float2bin, bin2float
from decimal import *
getcontext().prec = 64
from bitstring import BitArray
#import deepCABAC
import torch
#import torchac
import subprocess
#import glymur as gm
import numpy as np
import imageio
import yaml
import json
import gzip
import os
import io


def probability_est_jc(jacobian):
	soft_jc = torch.nn.Softmax(dim=-1)

def probability_est_kp(kp):
	soft_kp = torch.nn.Softmax(dim=-1)
	pmf = soft_kp(kp)
	print(pmf)
	return pmf
	

def compress(inp, bitout, freq = None):
	if freq:
		initfreqs = freq
	else:	
		initfreqs = FlatFrequencyTable(257)
	freqs = SimpleFrequencyTable(initfreqs)
	enc = ArithmeticEncoder(32, bitout)
	while True:
		# Read and encode one byte
		symbol = inp.read(1)
		if len(symbol) == 0:
			break
		enc.write(freqs, symbol[0])
		freqs.increment(symbol[0])
	enc.write(freqs, 256)  # EOF
	enc.finish()  # Flush remaining code bits
	return freqs
	
def decompress(bitin):
	'''
	Calls the arithmetic decompression routine
	'''
	initfreqs = FlatFrequencyTable(257)
	freqs = SimpleFrequencyTable(initfreqs)
	dec = ArithmeticDecoder(32, bitin)
	decoded = bytes(b'')
	while True:
		# Decode and write one byte
		symbol = dec.read(freqs)
		if symbol == 256:  # EOF symbol
			break
		#out.write(bytes((symbol,)))
		decoded = decoded.join(bytes((symbol,)))
		freqs.increment(symbol)
	return decoded



    
def quantize(kp_driving,bits = 16, seq=False):
	if seq:
		kp_list = []
		meta = []
		for kp in kp_driving:
			quant = quantize_tensor(kp ,num_bits=bits)
			quant_kp = quant.tensor.cpu().detach().numpy()
			scale_kp = quant.scale.item()
			zero_point = quant.zero_point
			kp_list.append(quant_kp)
			meta.append((scale_kp, zero_point))
		return kp_list, meta
	else:
		kp_norm_md = {}
		kp = {}
		#conversion to 16-bit integer from 32-bit floating point
		kp_norm_md["value"] = quantize_tensor(kp_driving["value"],num_bits=16)
		kp_norm_md["jacobian"] = quantize_tensor(kp_driving["jacobian"])
		#Extract data
		keypoints = kp_norm_md["value"].tensor.cpu().detach().numpy()
		jacobians = kp_norm_md["jacobian"].tensor.cpu().detach().numpy()
		kp['value'] = keypoints
		kp['jacobian'] = jacobians
	
		scale_kp = kp_norm_md["value"].scale.item()
		scale_jc = kp_norm_md["jacobian"].scale.item()
		zero_point_kp = kp_norm_md["value"].zero_point
		zero_point_jc = kp_norm_md["jacobian"].zero_point
		return kp, (scale_kp,scale_jc, zero_point_kp,zero_point_jc)
    
    
def create_directories(root, filename=None):
		'''Creates directories for the associated output files and data'''
		if filename==None:
			result_video_path = root+'/video'
			result_metrics_path = root+'/metrics'
			bitstream_path = root+ "/bitstream"
			try:
				os.mkdir(root)
			except FileExistsError:
				pass
			try:
				os.mkdir(root+"/decoded")
				os.mkdir(result_video_path)
				os.mkdir(result_metrics_path)
			except FileExistsError:
				pass
			try:
				os.mkdir(bitstream_path)
				os.mkdir(bitstream_path + '/moov')
			except FileExistsError:
				pass
		else:
			result_video_path = root+filename+'/video'
			result_metrics_path = root+filename+'/metrics'
			bitstream_path = root+filename+ "/bitstream"
			try:
				os.makedirs(result_video_path)
			except FileExistsError:
				pass
			try:
				os.makedirs(result_metrics_path)
			except FileExistsError:
				pass
			try:
				os.makedirs(bitstream_path)
			except FileExistsError:
				pass			
		return result_video_path, result_metrics_path, bitstream_path
		
		
def bitrate(total_bytes,fps,frames):
	#computes the encoding bitrate
	seq_bitrate = (total_bytes*8*fps)/(1000*frames)
	return seq_bitrate
	
	
def find_nearest_hevc(hevc_bitrates, dac_bitrate):
	#need to be more robust?
	qp = [20,25,26,28,30,32,33,35,37,38,39,40,43,45,50] #[30,35,40,45,50] #[20,25,30,33,35,38,40,43,45,50]
	array = np.asarray(hevc_bitrates)
	idx = (np.abs(array - dac_bitrate)).argmin()
	return qp[idx], idx
	
	
def pixel_loss(kp_source, kp_driving, kp=True):
	'''returns eucledian distance between two keypoints(1) or jacobians(0)'''
	if kp:
		diff = np.subtract(kp_source['value'].cpu().detach().numpy().flatten(), kp_driving['value'].cpu().detach().numpy().flatten() )
		sq_diff = [i**2 for i in diff]
		loss = np.sqrt(np.sum(sq_diff))
	else:
		diff = np.subtract(kp_source['jacobian'].cpu().detach().numpy().flatten(), kp_driving['jacobian'].cpu().detach().numpy().flatten() )
		sq_diff = [i**2 for i in diff]
		loss = np.sqrt(np.sum(sq_diff))
	return loss
	
def save_video(composite_video, filename, fps=30):
	imageio.mimsave(filename, [img_as_ubyte(frame) for frame in composite_video], fps=fps)


def euc_distance(source, target, kp=False):
	if kp:
		source = source.cpu().detach().numpy().flatten()
		target = target.cpu().detach().numpy().flatten()
	else:
		source = source[:,:,0]
		target = target[:,:,0]
	return np.sqrt(np.sum((source-target)**2))
	


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'], strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=False)
    
    #if not cpu:
    #    generator = DataParallelWithCallback(generator)
    #    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector, config
    
def pmf_to_cdf(pmf):
  cdf = torch.cumsum(pmf,dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  # On GPU, softmax followed by cumsum can lead to the final value being 
  # slightly bigger than 1, so we clamp.
  cdf_with_0 = cdf_with_0.clamp(max=1.)
  return cdf_with_0
  
def encode_exp(arr, order=2):
	code = []
	for num in arr:
		if num<0:
			#map to an even integer 2*num
			quo = ((2*abs(num))+2**order)-1
		else:
			#map to an odd integer 2*num-1
			quo = ((2*num-1)+2**order)-1
		n_b = bin(quo+1)
		q = n_b[2:]
		n_0 = len(n_b[2:])
		code.append('0'*n_0+q)
	code_str = ''.join(code)
	freq = {'0': code_str.count('0'),'1': code_str.count('1')}
	return code_str,freq
  
def exp_encoder(arr=None, order=2):
    stream = []
    for num in arr:
        code = encode_exp(num)
        stream.append(code)
    out = ''.join(stream)
    freq = {'0': out.count('0'),'1':out.count('1')}
    return out,freq
    
def encode_exp(num, order=2):
    if num<0:
        #map to an even integer 2*num
        quo = ((2*abs(num))+2**order)-1
    else:
        #map to an odd integer 2*num-1
        quo = ((2*num-1)+2**order)-1
    n_b = bin(quo+1)
    q = n_b[2:]
    n_0 = len(n_b[2:])
    code = '0'*n_0+q
    return code
    
def decode_exp(code, order=2):
    n = len(code.split('1')[0])
    n_b = '0b'+code[n:]
    quo = int(n_b, 2)-1
    num = (quo+1)-2**order
    if num%2 == 0:
        res = int(-num/2)
    else:
        res = int((num+1)/2)
    return res

def exp_decoder(code, order=2):
    arr = []
    temp_code = code
    while True:
        if len(temp_code)>0:
            n = len(temp_code.split('1')[0])
            num = decode_exp(temp_code[:2*n])
            arr.append(num)
            temp_code = temp_code[2*n:]
        else:
            break
    return np.array(arr)
    
def binarize(arr=None,bits=8):
	code = []
	for num in arr:
		word = np.binary_repr(num, width=bits)
		code.append(word)
	code = ''.join(code)
	freq = {'0':code.count('0'),'1': code.count('1')}
	return code, freq
    
#def bitstring_to_bytes(s):
#    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

def bitstring_to_bytes(s):
    #split to 8 bit chunks and conver to bytes
    byte_string = b''
    for n in range(0,len(s),8):
        num = int(s[n:n+8],2).to_bytes(1,'big')
        byte_string+= num
    return byte_string
    
def bytes_to_bitstring(byte_arr):
    nums = []
    for byte in byte_arr:
        n = bin(byte)[2:]
        if len(n)<8:
            diff = 8-len(n)
            n = '0'*diff+n 
            nums.append(n)
        else:
            nums.append(n)
    return ''.join(nums)
'''
    
def bytes_to_bitstring(byte_arr):
    length = len(byte_arr)*8-1
    bit_string = BitArray(bytes=byte_arr, length=length, offset=1)
    bits = bit_string.bin[3:]
    return bits
'''
		
def float2bin(num,bits=128):
    integral_part = int(num)
    binary = bin(integral_part)[2:]+'.'
    decimal_part = num-Decimal(integral_part)
    for idx in range(bits):
        fract = decimal_part*2
        if fract>1:
            binary+='1'
            decimal_part=fract-1
        elif fract==1.0:
            binary+='1'
            break
        else:
            binary+='0'
            decimal_part = fract
    return binary

def bin2float(num):
    result = Decimal(0.0)
    integral, fractional = num.split('.')
    for idx,bit in enumerate(integral):
        if bit == '0':
            continue
        mul = 2**idx
        result = result+ Decimal(mul)
    
    for idx, bit in enumerate(fractional):
        if bit == '0':
            continue
        mul = Decimal(1.0)/Decimal(2**(idx+1))
        result = result + mul
    return result
      

def write_nalu(bitstream,residual = None, source_img = None, kp=None,ref=0, kp_ref=False, nalu=0, moov_box=None, mdat_box = None,qp=50, prev_freq=None, frames=None):
	bytes = 0
	frame = []
	
	if moov_box is not None:
		with open(bitstream + "/mdat.json", "w") as mdat:
			mdat.write(json.dumps(moov_box))
			bytes += len(json.dumps(moov_box))
		
	if source_img is not None:
		encoder = BPGEncoder(qp=qp)
		bytes += encoder.encode(frame=source_img, output=bitstream+ "/" +str(nalu))
			
	if mdat_box is not None:
		with open(bitstream + "/qp.json", "w") as mdat:
			mdat.write(json.dumps(mdat_box))
		bytes += len(json.dumps(mdat_box))
			
	if kp is not None:
		#getcontext().prec = 128
		#freqs = 0
		#kps = kp
		#print(len(kps))
		#detect zeros and signal to use the last frame kps
		
		#print('Original:', kps)
		#code, freqs = exp_encoder(arr=kps)
		#code_bytes  = bitstring_to_bytes(code)
		#with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
		#	keypoints.write(kps)
		#bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		#code_h = np.frombuffer(kps, np.int8)
		#dec_kp = exp_decoder(code_h)
		#print('Decoded: ',code_h)
		
		
		#n_frames = len(kp)
		#print(n_frames)
		#shape = kp[0].shape
		#if frames:
		#	byte_arr = frames.to_bytes(1,'big') + np.array(kp).tobytes()
		#else:
		#	byte_arr = np.array(kp).tobytes()
		'''
                if not kp_ref:
			with open(bitstream + "/"  + str(nalu)+ "_kp_in.bin", "wb") as keypoints:
				keypoints.write(kp)
			raw_bytes = os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp_in.bin")
			print('Raw: ', raw_bytes)
			if prev_freq == None:
				initfreqs = FlatFrequencyTable(257)
				freqs = SimpleFrequencyTable(initfreqs)
			else:
				freqs  = SimpleFrequencyTable(prev_freq)
			with open(bitstream + "/"  + str(nalu)+ "_kp_in.bin", 'rb') as inp, contextlib.closing(BitOutputStream(open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb"))) as bitout:
				enc = ArithmeticEncoder(32,bitout)
				while True:
					# Read and encode one byte
					symbol = inp.read(1)
					if len(symbol) == 0:
						break
					enc.write(freqs, symbol[0])
					freqs.increment(symbol[0])
				enc.write(freqs, 256)  # EOF
				enc.finish()	
			bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
			print('Compressed bytes: ', bytes)
			os.remove(bitstream + "/"  + str(nalu)+ "_kp_in.bin")
		else:
			with open(bitstream + "/"  + str(nalu)+ "_kp_ref.bin", "wb") as keypoints:
				keypoints.write(kp)
			bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp_ref.bin")
		
		
		
		'''
                '''
		print('code length: ',len(code))
		msg_len = len(code).to_bytes(1,'big')
		freq_0 = freq['0'].to_bytes(1,'big')
		freq_1 = freq['1'].to_bytes(1,'big')
		print(code)
		#arithmetic coding
		ae_enc = ArithmeticEncoding(frequency_table=freq)
		probs = ae_enc.probability_table
		
		encoded_msg, encoder , interval_min_value, interval_max_value = ae_enc.encode(msg=code,probability_table=probs)
		
		#print('Encoded: ',encoded_msg)
		byte_msg = msg_len+freq_0+freq_1+np.float64(encoded_msg).tobytes()
		#print('Encoded bytes: ', byte_msg)
		
		#print(freq, len(code))
		#binary_arr = bitstring_to_bytes(code)
		#print(binary_arr)
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
			keypoints.write(byte_msg)
		bytes = os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		#print('Encoded size: ', bytes,' Bytes')
		
		print('---------Decoding-----------')
		msg_len = np.frombuffer(byte_msg[0:1], dtype=np.int8)[-1]
		freq_0 = np.frombuffer(byte_msg[1:2], dtype=np.int8)[-1]
		freq_1 = np.frombuffer(byte_msg[2:3],dtype=np.int8)[-1]
		msg_h = np.frombuffer(byte_msg[3:], dtype=np.float64)[-1]
		freq = {'0': freq_0, '1':freq_1}
		ae_dec = ArithmeticEncoding(frequency_table=freq)
		probs = ae_dec.probability_table
		decoded_msg, decoder = ae_dec.decode(encoded_msg=msg_h, msg_length=msg_len,probability_table=probs)
		print(''.join(decoded_msg))
		freqs = 0
		'''
		
		'''
		kps = (kp).cpu().detach().numpy().reshape(20,).astype(np.int8)
		
		#jcs = np.round(((kp['jacobian']/0.01)+127).cpu().detach().numpy().reshape(40,).astype(np.uint8))
		arr = list(kps) #+list(jcs)
		codes = []
		freqs =[]
		start, stop = 0, 5
		ref_frame = b''+ref.to_bytes(1,'big') #1-byte for reference frame,
		#separate x and y coordinates for stronger correlation coding
		#code, freq = binarize(arr= arr, bits=8)
		code, freq = exp_encoder(arr=arr)
		print(len(code))
		'''
		'''
		#print('-----------------------------------------')
		ae_enc = ArithmeticEncoding(frequency_table=freq)
		probs = ae_enc.probability_table
		dist_0 = int(probs['0']*100).to_bytes(1,'big')
		dist_1 = len(code).to_bytes(2,'big')
		encoded_chunk, encoder , interval_min_value, interval_max_value = ae_enc.encode(msg=code,probability_table=ae_enc.probability_table)
		print(encoded_chunk)
		binary = float2bin(encoded_chunk, 100)
		dec_msg = bin2float(binary)
		#byte_arr = np.float32(encoded_chunk).tobytes()
		#print(byte_arr)
		
		#dec_msg = np.frombuffer(byte_arr, dtype=np.float32)[-1]
		print(dec_msg)
		decoded_msg, decoder = ae_enc.decode(encoded_msg=dec_msg, msg_length=len(code),probability_table=ae_enc.probability_table)
		print(''.join(decoded_msg))
		'''
		'''
		binary = float2bin(encoded_chunk, 64)
		print(binary)
		print('-------------------------------------')
		print(bin2float(binary))
		binary_arr = bitstring_to_bytes(binary[2:])
		byte_arr = dist_0 +dist_1+ binary_arr
		print(byte_arr)
		'''
		'''
		#storage
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
			keypoints.write(byte_arr)
		bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		#decoding
		
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "rb") as data:
			kp_byte_stream = data.read()
		prob_0 = np.frombuffer(kp_byte_stream[0:1], dtype=np.uint8)[-1]
		probs_dec = {'0': prob_0/100, '1': 1-(prob_0)/100}
		print(probs_dec)
		'''
		'''
		dec_binary = '0b0.'+ bytes_to_bitstring(kp_byte_stream[1:])
		msg_hat = bin2float(dec_binary)
		print(msg_hat)
		ae_dec = ArithmeticEncoding(probability_table=probs_dec, encoder=False)
		decoded_msg, decoder = ae_dec.decode(encoded_msg=msg_hat, msg_length=len(code),probability_table=ae_dec.probability_table)
		dec_msg_hat = ''.join(decoded_msg)
		print(dec_msg_hat)
		'''
		'''
		#print('-----------------------------------------')
		#msg = encoded_chunk.tobytes()
		#print(msg)
		#dec_msg = np.frombuffer(msg, dtype=np.float128)[-1]
		#print(dec_msg)
		#encode and decode
		#decoded_msg, decoder = ae_enc.decode(encoded_msg=dec_msg, msg_length=len(code),probability_table=ae_enc.probability_table)
		#dec_msg_hat = ''.join(decoded_msg)
		#print(dec_msg_hat)
		
		while start<len(arr):
			if stop > len(arr):
				code, freq = exp_encoder(arr= arr[start:])
			else:
				code, freq = exp_encoder(arr= arr[start:stop])
			#initialize AE
			ae_enc = ArithmeticEncoding(frequency_table=freq)
			#encode the values
			probs = ae_enc.probability_table
			#print(probs)
			encoded_chunk, encoder , interval_min_value, interval_max_value = ae_enc.encode(msg=code,probability_table=ae_enc.probability_table)
			#convert chunk to byte stream
			chunk = np.float16(encoded_chunk).tobytes()
			#add meta information::  1 byte for original chunk length in bits, 1 byte for probability dist
			length = len(code).to_bytes(1,'big')
			dist_0 = freq['0'].to_bytes(1,'big')
			dist_1 = freq['1'].to_bytes(1,'big')
			byte_stream+= length+dist_0+dist_1+chunk
			stop+=5
			start+=5
			codes.append(code)
			freqs.append(freq)
			if start>len(arr):
				break
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
			keypoints.write(byte_stream)
		bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		'''
		'''
		
		'''
		'''
		kps = kp.cpu().detach().numpy().astype(np.int8).reshape(20,)
		#print(kps)
		#exp-golomb coding
		code, freq = exp_encoder(arr=kps[:10], order=2)
		print('code length: ',len(code))
		msg_len = len(code).to_bytes(1,'big')
		freq_0 = freq['0'].to_bytes(1,'big')
		freq_1 = freq['1'].to_bytes(1,'big')
		print(code)
		#arithmetic coding
		ae_enc = ArithmeticEncoding(frequency_table=freq)
		probs = ae_enc.probability_table
		
		encoded_msg, encoder , interval_min_value, interval_max_value = ae_enc.encode(msg=code,probability_table=probs)
		
		#print('Encoded: ',encoded_msg)
		byte_msg = msg_len+freq_0+freq_1+np.float64(encoded_msg).tobytes()
		#print('Encoded bytes: ', byte_msg)
		
		#print(freq, len(code))
		#binary_arr = bitstring_to_bytes(code)
		#print(binary_arr)
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
			keypoints.write(byte_msg)
		bytes = os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		#print('Encoded size: ', bytes,' Bytes')
		
		print('---------Decoding-----------')
		msg_len = np.frombuffer(byte_msg[0:1], dtype=np.int8)[-1]
		freq_0 = np.frombuffer(byte_msg[1:2], dtype=np.int8)[-1]
		freq_1 = np.frombuffer(byte_msg[2:3],dtype=np.int8)[-1]
		msg_h = np.frombuffer(byte_msg[3:], dtype=np.float64)[-1]
		freq = {'0': freq_0, '1':freq_1}
		ae_dec = ArithmeticEncoding(frequency_table=freq)
		probs = ae_dec.probability_table
		decoded_msg, decoder = ae_dec.decode(encoded_msg=msg_h, msg_length=msg_len,probability_table=probs)
		print(''.join(decoded_msg))
		freqs = 0
		'''
		'''
		#byte_arr  = np.round(((np.array(kp)/0.01)).astype(np.int8)).tobytes()
		byte_arr = kp.cpu().detach().numpy().astype(np.int8).tobytes()
		
		if frames != None:
			nalus = frames.to_bytes(2,'big')
			byte_arr = nalus+ byte_arr
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
			keypoints.write(byte_arr)
		bytes = os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		print('Raw: ', bytes)
		'''
		'''
		if prev_freq == None:
			initfreqs = FlatFrequencyTable(10000)
			freqs = SimpleFrequencyTable(initfreqs)
		else:
			freqs  = SimpleFrequencyTable(prev_freq)
		with open(bitstream + "/"  + str(nalu)+ "_kp_in.bin", 'rb') as inp, contextlib.closing(BitOutputStream(open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb"))) as bitout:
			enc = ArithmeticEncoder(32,bitout)
			while True:
				# Read and encode one byte
				symbol = inp.read(2)
				if len(symbol) == 0:
					break
				enc.write(freqs, symbol[0])
				freqs.increment(symbol[0])
			enc.write(freqs, 9999)  # EOF
			enc.finish()	
		bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		os.remove(bitstream + "/"  + str(nalu)+ "_kp_in.bin")
		'''
		
		
		
		
		'''
		code_2, freq_2 = exp_encoder(arr= arr[10:])
		ae_enc = ArithmeticEncoding(frequency_table=freq_1)
		print(code_1)
		encoded_msg, encoder , interval_min_value, interval_max_value = ae_enc.encode(msg=code_1,probability_table=ae_enc.probability_table)
		msg = np.float16(encoded_msg).tobytes()
		print('Bytes: ',len(msg))
		print(ae_enc.probability_table)
		'''
		'''
		with open(bitstream + "/"  + str(nalu)+ "_kp_in.bin", "wb") as keypoints:
			keypoints.write(msg)
		size = os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp_in.bin")
		print(size)
		
		decoded_msg, decoder = ae_enc.decode(encoded_msg=encoded_msg, msg_length=len(code_1),probability_table=ae_enc.probability_table)
		dec_msg = ''.join(decoded_msg)
		print(dec_msg)
		
		
		
		#binary_code, encoder_binary = ae_enc.encode_binary(float_interval_min=interval_min_value,float_interval_max=interval_max_value)
		#print(binary_code)
		#print('P: ',len(code), 'bits')
		#print(freq)
		freqs = None
	
		kps = np.round(((kp['value']/0.01)+127).cpu().detach().numpy().reshape(20,).astype(np.uint8)).tobytes()
		jcs = np.round(((kp['jacobian']/0.01)+127).cpu().detach().numpy().reshape(40,).astype(np.uint8)).tobytes()
		byte_array = b''.join([kps,jcs])
		with open(bitstream + "/"  + str(nalu)+ "_kp_in.bin", "wb") as keypoints:
			keypoints.write(byte_array)
		#with open(bitstream + "/"  + str(nalu)+ "_jc_in.bin", "wb") as jacobians:
		#	jacobians.write(jcs)
		if prev_freq == None:
			initfreqs = FlatFrequencyTable(257)
			freqs = SimpleFrequencyTable(initfreqs)
		else:
			freqs  = SimpleFrequencyTable(prev_freq)
		with open(bitstream + "/"  + str(nalu)+ "_kp_in.bin", 'rb') as inp, contextlib.closing(BitOutputStream(open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb"))) as bitout:
			enc = ArithmeticEncoder(32,bitout)
			while True:
				# Read and encode one byte
				symbol = inp.read(1)
				if len(symbol) == 0:
					break
				enc.write(freqs, symbol[0])
				freqs.increment(symbol[0])
			enc.write(freqs, 256)  # EOF
			enc.finish()	
		bytes = os.path.getsize(bitstream + "/"  + str(nalu)+ "_kp.bin")
		print('P: ', bytes)
		
		interv = 0.1
		#if kp_ref:
		stepsize = 2 ** (-0.5*15)
		_lambda = 0.001
		
		kp_encoder  = deepCABAC.Encoder()
		kps = torch.unsqueeze(kp['value'],0).cpu().detach().numpy().astype(np.float16) #.reshape(20,)
		#encode keypoints
		kp_encoder.encodeWeightsRD(kps, interv, stepsize, _lambda )
		kp_stream = kp_encoder.finish()
		bytes+=kp_stream.size
		with open(bitstream + "/"  + str(nalu)+ "_kp.bin", "wb") as keypoints:
			keypoints.write(kp_stream)
		
	
		#encode jacobain matrix		
		jc_encoder  = deepCABAC.Encoder()
		jc = kp['jacobian'].cpu().detach().numpy().astype(np.float16)
		jc_encoder.encodeWeightsRD(jc, interv, stepsize, _lambda )
		jc_stream = jc_encoder.finish()
		bytes += jc_stream.size
		with open(bitstream + "/"  + str(nalu)+ "_jc.bin", "wb") as jacobian:
			jacobian.write(jc_stream)
		print('P {}: {} bytes'.format(nalu, bytes))	
		'''
		pass

	if residual is not None:
		with gzip.GzipFile(bitstream + "/"  + str(nalu)+ "_res.npy.gz", "wb") as res:
			np.save(file= res, arr = np.array(residual))
		bytes += os.path.getsize(bitstream + "/"  + str(nalu)+ "_res.npy.gz")		
	if kp is not None and not kp_ref:
		return bytes, freqs
	else:
		return bytes
	
def read_nalu():
	pass
	
def video_writer(predictions,result_video_path, filename = 'decoded_video', visualization = None, fps = 10):
	if visualization == None:
		imageio.mimsave(result_video_path+ '/' +filename+".mp4", [img_as_ubyte(frame) for frame in predictions], fps=fps)  
	else:
		out_video = []
		for idx in range(len(predictions)):
			composite_frame = np.concatenate((visualization[idx],img_as_ubyte(predictions[idx])), axis=1)
			out_video.append(composite_frame)
		imageio.mimsave(result_video_path+ '/' +filename+"_vis.mp4", [img_as_ubyte(frame) for frame in out_video], fps=fps)
