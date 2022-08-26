import os, abc, time, sys
import torch
import imageio
import subprocess
import numpy as np
import PIL.Image as Image
from tempfile import mkstemp
from pytorch_msssim import ms_ssim

from typing import Dict, List, Optional, Union
# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return Image.open(filepath).convert(mode)

def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def _compute_ms_ssim(a, b, max_val: float = 255.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()


_metric_functions = {
    "psnr": _compute_psnr,
    "ms-ssim": _compute_ms_ssim,
}

def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    metrics: Optional[List[str]] = None,
    max_val: float = 255.0,
) -> Dict[str, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`."""

    if metrics is None:
        metrics = ["psnr"]

    def _convert(x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x

    a = _convert(a)
    b = _convert(b)

    out = {}
    for metric_name in metrics:
        out[metric_name] = _metric_functions[metric_name](a, b, max_val)
    return out


def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def _get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4]

class Codec(abc.ABC):
    """Abstract base class"""

    _description = None

    @property
    def description(self):
        return self._description

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    @abc.abstractmethod
    def _run_impl(self, img, quality, *args, **kwargs):
        raise NotImplementedError()

    def run(self,in_filepath,quality: int,metrics: Optional[List[str]] = None,return_rec: bool = False,):
        if isinstance(in_filepath, np.ndarray):
            #create a temporary file for the input image
            fd_in, png_in_filepath = mkstemp(suffix=".png")
            imageio.imsave(png_in_filepath, in_filepath)
            in_file = png_in_filepath
            #compression
            info, rec = self._run_impl(in_file, quality)
            info.update(compute_metrics(rec, self._load_img(in_file), metrics))
            os.close(fd_in)
            os.remove(png_in_filepath)
            
        else:
            in_file = in_filepath
            info, rec = self._run_impl(in_file, quality)
            info.update(compute_metrics(rec, self._load_img(in_file), metrics))
        return info


class BinaryCodec(Codec):
    """Call an external binary."""

    fmt = None

    @property
    def name(self):
        raise NotImplementedError()

    def _run_impl(self, in_filepath, quality):
        fd0, png_filepath = mkstemp(suffix=".png")
        fd1, out_filepath = mkstemp(suffix=self.fmt)

        # Encode
        start = time.time()
        run_command(self._get_encode_cmd(in_filepath, quality, out_filepath))
        enc_time = time.time() - start
        size = filesize(out_filepath)
        # Decode
        start = time.time()
        run_command(self._get_decode_cmd(out_filepath, png_filepath))
        dec_time = time.time() - start

        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)
        os.close(fd1)
        os.remove(out_filepath)

        img = self._load_img(in_filepath)
        bpp_val = float(size) * 8 / (img.size[0] * img.size[1])
        vis_img = np.concatenate((img, rec), axis=1)

        out = {
            "bpp": bpp_val,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            'bit_size': size,
            'vis': vis_img,
            'rec': np.array(rec)
        }

        return out, rec

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        raise NotImplementedError()

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        raise NotImplementedError()


class BPG(BinaryCodec):
    """BPG from Fabrice Bellard."""
    def __init__(self, color_mode="rgb", encoder="x265",
                        subsampling_mode="420", bit_depth='8', 
                        encoder_path='bpgenc', decoder_path='bpgdec'):
        self.fmt = ".bpg"
        self.color_mode = color_mode
        self.encoder = encoder
        self.subsampling_mode = subsampling_mode
        self.bitdepth = bit_depth
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path



    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.encoder_path)}"



    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 0 <= quality <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            in_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd


if __name__ == "__main__":
    img_n = 8
    img = f"imgs/{img_n}.png"
    img_arr = imageio.imread(img)
    qp = 30
    bpg = BPG()

    
    #pass img with file path
    out  = bpg.run(img,qp, ['psnr','ms-ssim'])
    imageio.imsave(f"rec/{img_n}_{qp}_vis.png", out['vis'])

    #pass img as np.ndarray :: #Images must be uint8
    out  = bpg.run(img_arr,qp, ['psnr','ms-ssim'])
    imageio.imsave(f"rec/{img_n}_{qp}_vis_arr.png", out['vis'])