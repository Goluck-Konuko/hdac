from pathlib import Path
import struct
import os
import torch
import numpy as np
import contextlib
from tempfile import mkstemp
from typing import List
from .arithmetic_coder import ArithmeticEncoder,FlatFrequencyTable,SimpleFrequencyTable,ArithmeticEncoder,BitOutputStream
import itertools

class KeypointCompress:
    def __init__(self,q_step=64, num_kp=10, out_dir = 'log/') -> None:
        self.q_step = q_step
        self.num_kp = num_kp
        self.residuals = []
        self.out_dir = out_dir
        self.total_bytes = 0
        self.idx = 1
        self.kp_reference = None

    def reset(self):
        self.kp_reference = None
        self.total_bytes = 0
        self.residuals = []

    def _get_residual(self,src,drv):
        sr = torch.round((src+1)*self.q_step/1).int()
        dr = torch.round((drv+1)*self.q_step/1).int()
        val_diff = sr-dr
        return val_diff.flatten().tolist()

    def _decode_kp_val(self, src, kp_res):
        kp_res = torch.tensor(np.array(kp_res).reshape(1, self.num_kp, 2), dtype=torch.float32)
        sr = torch.round((src+1)*self.q_step/1).int()
        dr = (sr - kp_res)/self.q_step - 1
        return dr

    def _decode_kp_jacobians(self,src, kp_res):
        kp_res = torch.tensor(np.array(kp_res).reshape(1, self.num_kp, 2,2), dtype=torch.float32)
        dr = (src - kp_res)/self.q_step - 1
        return dr

    def encode_kp(self,kp_source: dict,kp_driving: dict):
        if self.kp_reference is None:
            self.kp_reference = kp_source
        kp_res = self._get_residual(self.kp_reference['value'],kp_driving['value'])
        res = kp_res
        if 'jacobian' in kp_driving:
            jc_res = self._get_residual(self.kp_reference['jacobian'],kp_driving['jacobian'])
            res += jc_res

        #decode the kps for animation
        kp_val = self._decode_kp_val(self.kp_reference['value'], res[:self.num_kp*2])
        kp_frame     = {'value': kp_val}
        if 'jacobian' in kp_driving:
            kp_jacobian = self._decode_kp_jacobians(self.kp_reference['jacobian'], res[self.num_kp*2:])
            kp_frame['jacobian'] = kp_jacobian
        self.residuals.append(res)
        out_path=self.out_dir+f'_{self.idx}.bin'
        final_encoder_expgolomb(res,out_path, 0) 
        self.total_bytes += filesize(out_path)
        os.remove(out_path)
        self.kp_reference = kp_frame
        # total_bytes += write_kp_bitstream(self.residuals, out_path=self.out_dir+f'_{self.idx}.bin', order=0)
        return kp_frame, res

    def get_bitstream(self):
        # bytes = write_kp_bitstream(self.residuals, out_path=self.out_dir, order=0)
        return self.total_bytes*8



###0-order
def get_digits(num):
    result = list(map(int,str(num)))
    return result

def exponential_golomb_encode(n):
    unarycode = ''
    golombCode =''
    ###Quotient and Remainder Calculation
    groupID = np.floor(np.log2(n+1))
    temp_=groupID
    while temp_>0:
        unarycode = unarycode + '0'
        temp_ = temp_-1
    unarycode = unarycode#+'1'

    index_binary=bin(n+1).replace('0b','')
    golombCode = unarycode + index_binary
    return golombCode
        

def exponential_golomb_decode(golombcode):
    code_len=len(golombcode)
    m= 0 ### 
    for i in range(code_len):
        if golombcode[i]==0:
            m=m+1
        else:
            ptr=i  ### first 0
            break

    offset=0
    for ii in range(ptr,code_len):
        num=golombcode[ii]
        offset=offset+num*(2**(code_len-ii-1))
    decodemum=offset-1
    
    return decodemum


def expgolomb_split(expgolomb_bin_number):
    x_list=expgolomb_bin_number
    
    del(x_list[0])
    x_len=len(x_list)
    
    sublist=[]
    while (len(x_list))>0:

        count_number=0
        i=0
        if x_list[i]==1:
            sublist.append(x_list[0:1])
            del(x_list[0])            
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            count_number=count_number+num_times_zeros[0]
            sublist.append(x_list[0:(count_number*2+1)])
            del(x_list[0:(count_number*2+1)])
    return sublist

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def dataconvert_expgolomb(symbol):
    for i in range(len(symbol)):
        if symbol[i] <=0:
            symbol[i]=(-symbol[i])*2
        else:
            symbol[i]=symbol[i]*2-1

    return symbol

def list_binary_expgolomb(symbol):
    for i in range(len(symbol)):
        n = symbol[i]
        symbol[i]=exponential_golomb_encode(n)

    m='1'
    for x in symbol:
        m=m+str(x)
    return m


class PpmModel:
	def __init__(self, order, symbollimit, escapesymbol):
		if order < -1 or symbollimit <= 0 or not (0 <= escapesymbol < symbollimit):
			raise ValueError()
		self.model_order = order
		self.symbol_limit = symbollimit
		self.escape_symbol = escapesymbol
		
		if order >= 0:
			self.root_context = PpmModel.Context(symbollimit, order >= 1)
			self.root_context.frequencies.increment(escapesymbol)
		else:
			self.root_context = None
		self.order_minus1_freqs = FlatFrequencyTable(symbollimit)  ###arithmeticcoding.
	
	
	def increment_contexts(self, history, symbol):
		if self.model_order == -1:
			return
		if len(history) > self.model_order or not (0 <= symbol < self.symbol_limit):
			raise ValueError()
		
		ctx = self.root_context
		ctx.frequencies.increment(symbol)
		for (i, sym) in enumerate(history):
			subctxs = ctx.subcontexts
			assert subctxs is not None
			
			if subctxs[sym] is None:
				subctxs[sym] = PpmModel.Context(self.symbol_limit, i + 1 < self.model_order)
				subctxs[sym].frequencies.increment(self.escape_symbol)
			ctx = subctxs[sym]
			ctx.frequencies.increment(symbol)
	
	
	
	# Helper structure
	class Context:
		
		def __init__(self, symbols, hassubctx):
			self.frequencies = SimpleFrequencyTable([0] * symbols) ##arithmeticcoding.
			self.subcontexts = ([None] * symbols) if hassubctx else None

def encode_symbol(model, history, symbol, enc):
    # Try to use highest order context that exists based on the history suffix, such
    # that the next symbol has non-zero frequency. When symbol 256 is produced at a context
    # at any non-negative order, it means "escape to the next lower order with non-empty
    # context". When symbol 256 is produced at the order -1 context, it means "EOF".
    for order in reversed(range(len(history) + 1)):
        #print(order)
        ctx = model.root_context
        for sym in history[ : order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            if ctx is None:
                break
        else:# ctx is not None
            if symbol != 2 and ctx.frequencies.get(symbol) > 0: ##############
                enc.write(ctx.frequencies, symbol)
                return
            # Else write context escape symbol and continue decrementing the order
            enc.write(ctx.frequencies, 2) ##############
    # Logic for order = -1
    enc.write(model.order_minus1_freqs, symbol)


##### Decode symbol based PPM MODEL 

def decode_symbol(dec, model, history):
    # Try to use highest order context that exists based on the history suffix. When symbol 256
    # is consumed at a context at any non-negative order, it means "escape to the next lower order
    # with non-empty context". When symbol 256 is consumed at the order -1 context, it means "EOF".
    for order in reversed(range(len(history) + 1)):
        ctx = model.root_context
        for sym in history[ : order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            if ctx is None:
                break
        else:  # ctx is not None
            symbol = dec.read(ctx.frequencies)
            if symbol < 2: #############
                return symbol
                # Else we read the context escape symbol, so continue decrementing the order
    # Logic for order = -1
    return dec.read(model.order_minus1_freqs)

def write_exp_golomb(data, out_path, order=0):
     with contextlib.closing(BitOutputStream(open(out_path, "wb"))) as bitout:  
        enc = ArithmeticEncoder(256, bitout) #########arithmeticcoding.

        model = PpmModel(order, 3, 2)  ##########ppmmodel.
        history = []

        # Read and encode one byte
        symbol=data

        symbol = dataconvert_expgolomb(symbol)
        symbollist = list_binary_expgolomb(symbol)
             
        for ii in symbollist:
            i_number=int(ii)
            
            encode_symbol(model, history, i_number, enc)

            model.increment_contexts(history, i_number)
            if model.model_order >= 1:
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, i_number) ###########
            #print(history)
        encode_symbol(model, history, 2, enc)  # EOF ##########
        enc.finish()  #        

def final_encoder_expgolomb(datares,outputfile, MODEL_ORDER = 0):
     # Must be at least -1 and match ppm-decompress.py. Warning: Exponential memory usage at O(257^n).
    with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:  #arithmeticcoding.
    
        enc = ArithmeticEncoder(256, bitout) #########arithmeticcoding.
        #print(enc)
        model = PpmModel(MODEL_ORDER, 3, 2)  ##########ppmmodel.
        #print(model)
        history = []

        # Read and encode one byte
        symbol=datares
        #print(symbol)
        
        # 数值转换
        symbol = dataconvert_expgolomb(symbol)
        #print(symbol)
        symbollist = list_binary_expgolomb(symbol)

        for ii in symbollist:
            #print(ii)
            i_number=int(ii)
            
            encode_symbol(model, history, i_number, enc)

            model.increment_contexts(history, i_number)
            if model.model_order >= 1:
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, i_number) ###########
        encode_symbol(model, history, 2, enc)  # EOF ##########
        enc.finish()  # 

def write_kp_bitstream(kp_list, out_path='bitstream.bin', order=0):
    total_bytes = 0
    for idx, kp in enumerate(kp_list):
        final_encoder_expgolomb(kp,out_path, order)   
        total_bytes += filesize(out_path)
        os.remove(out_path)
    return total_bytes

# def write_kp_bitstream(kp_list, out_path='bitstream.bin'):
#     # kps = np.array(kp_list).tobytes()
#     #initialize arithmetic coding
#     #NOTE: Entropy coding still not optimized!
#     initfreqs = FlatFrequencyTable(257)
#     freqs  = None
#     total_bytes = 0
#     for idx, kp in enumerate(kp_list):
#         kp_in, kp_in_filepath = mkstemp(suffix=".bin")
#         kps = np.array(kp).tobytes()

#         with open(kp_in_filepath, "wb") as keypoints:
#             keypoints.write(kps)
					
#         if freqs is None:
#             freqs = FlatFrequencyTable(initfreqs)
#         else:
#             freqs = FlatFrequencyTable(freqs)
#         with open(kp_in_filepath, 'rb') as inp, contextlib.closing(BitOutputStream(open(out_path, "wb"))) as bitout:
#             enc = ArithmeticEncoder(32,bitout)
#             while True:
#                 # Read and encode one byte
#                 symbol = inp.read(1)
#                 if len(symbol) == 0:
#                     break
#                 enc.write(freqs, symbol[0])
#                 freqs.increment(symbol[0])
#             enc.write(freqs, 256)  # EOF
    #         enc.finish()

    #     os.close(kp_in)
    #     os.remove(kp_in_filepath)
    #     total_bytes += filesize(out_path)
    #     os.remove(out_path)
    # return total_bytes

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def write_bitstream(strings, out_path='bistream'):
    source_frame_bits = strings[0][0][0]
    y_string = strings[0][1][0]
    kps = strings[1:]
    n_kps = len(kps)
    with Path(out_path+'.bin').open('wb') as f:
        len_source = len(source_frame_bits)
        len_source_latents = len(y_string)
        write_uints(f, (len_source,len_source_latents, n_kps))
        write_bytes(f,source_frame_bits)
        write_bytes(f, y_string)
  
        for kp in kps:
            point = kp['kp_strings'][0]
            len_kp = len(point)
            jacobian = kp['jc_strings'][0]
            len_jc = len(jacobian)
            write_uints(f, (len_kp,len_jc))
            write_bytes(f, point)
            write_bytes(f, jacobian)
    return filesize(out_path+'.bin')