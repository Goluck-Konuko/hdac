# Ultra-low bitrate video compression using deep animation models

This repository contains the source code for the paper [A HYBRID DEEP ANIMATION CODEC FOR LOW-BITRATE VIDEO CONFERENCING](https://arxiv.org/abs/2207.13530) published in ICIP 2022


## Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

## Assets
### YAML Config
Describes the configuration settings for the pre-trained model. See ```config/dac.yaml```.
Use ```config/dac_test.yaml``` at inference time

### Pre-trained checkpoint
Checkpoints can be found under following link: [google-drive](https://drive.google.com/file/d/1haJ2vmp5RHS5MdSh2Pk7EnUNgiLQgPlS/view?usp=sharing). Download and place in the ```cpks/``` directory.

#### BPG Codec
Required for source frame compression under low latency conditions. Can be replaced with any off the shelf (Fast) image compression algorithm like JPEG or JPEG2000

#### HM-HEVC Codec
We include the standard HM-HEVC codec implementation for benchmark testing. Verify HEVC configuration and change as desired.

#### Metrics
We include a metrics module combining the suggestions from JPEG-AI with popular quantiative metrics used in computer vision and beyond.
Supported metrics: 'psnr','fsim','iw_ssim','ms_ssim','ms_ssim_pytorch','vif','nlpd', 'vmaf', 'psnr_hvs','vif','lpips', 'lpips_vgg','pim'

### Datasets
 	**VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

 	**Creating your own videos**. 
 	The input videos should be cropped to target the speaker's face at a resolution of 256x256 (Updates are underway to add higher resolution). 

 ## Video Compression
 	```
 	python run.py --config config/dac.yaml --checkpoint cpks/>checkpoint_name.pth.tar --start_frame START --end_frame END --gop_size GOP --output_dir OUTPUT_DIR --compute_metrics 

### CREDITS
This repository contains an adaptation of the source code for the paper [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation) by Siarohin et al. NeurIPS 2019
We optimize our keypoint entropy coding by extending the exp-golomb coding + Arithmetic coding inspired by the work of [Chen et al. DCC 2022](https://github.com/alibaba-edu/temporal-evolution-inference-with-compact-feature-representation-for-talking-face-video-compression)