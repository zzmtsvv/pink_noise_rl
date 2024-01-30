from typing import Tuple, Union, Optional, Iterable
import numpy as np
from numpy.fft import irfft, rfftfreq


class ColoredNoiseProcess:
    '''
    https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1995A%26A...300..707T&defaultprint=YES&page_ind=0&filetype=.pdf

    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    '''
    def __init__(self,
                 beta: float,
                 size: Union[int, Tuple[int]],
                 scale: int = 1,
                 max_period: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.beta = beta
        # white noise: beta = 0
        # red noise: beta = 2 (brownian motion, related to OU noise)

        min_freq = 0
        if max_period is not None:
            min_freq = 1 / max_period
        self.min_freq = min_freq

        self.scale = scale
        self.rng = rng

        if not isinstance(size, list):
            size = [size]
        self.size = size
        self.timesteps = self.size[-1]

        self.reset()
    
    def reset(self) -> None:
        self.buffer = self.powerlaw_gaussian(self.beta,
                                             self.size,
                                             self.min_freq,
                                             self.rng)
        self.idx = 0
    
    def sample(self, T: int = 1) -> np.ndarray:
        n = 0
        res = []
        while n < T:
            if self.idx >= self.timesteps:
                self.reset()
            m = min(T - n, self.timesteps - self.idx)
            res.append(self.buffer[..., self.idx:self.idx + m])
            n += m
            self.idx += m
        
        res = self.scale * np.concatenate(res, axis=-1)
        if n > 1:
            return res
        return res[..., 0]

    @staticmethod
    def powerlaw_gaussian(exponent: float,
                          size: Iterable[int],
                          min_freq: float = 0,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
        samples = size[-1]
        f = rfftfreq(samples)

        assert 0 <= min_freq <= 0.5, "Low frequency cutoff error"
        min_freq = max(min_freq, 1.0 / samples)

        scale = f
        cutoff_idx = np.sum(scale < min_freq)
        if cutoff_idx and cutoff_idx < scale.shape[0]:
            scale[:cutoff_idx] = scale[cutoff_idx]
        scale = np.power(scale, -exponent / 2.0)

        w = scale[1:].copy()
        w[-1] *= (1 + (samples % 2)) / 2.    # correct f = +-0.5
        sigma = 2 * np.sqrt(np.sum(np.square(w))) / samples

        size[-1] = len(f)

        dims_to_add = len(size) - 1
        scale = scale[(None,) * dims_to_add + (Ellipsis,)]

        if rng is None:
            rng = np.random.default_rng()
        scale_re = rng.normal(scale=scale, size=size)
        scale_im = rng.normal(scale=scale, size=size)

        if not (samples % 2):
            scale_im[..., -1] = 0
            scale_re[..., -1] *= np.sqrt(2)    # Fix magnitude

        scale_im[..., 0] = 0
        scale_re[..., 0] *= np.sqrt(2)    # Fix magnitude

        s = scale_re + 1J * scale_im
        y = irfft(s, n=samples, axis=-1) / sigma
        return y


class PinkNoiseProcess(ColoredNoiseProcess):
    def __init__(self,
                 size: Union[int, Tuple[int]],
                 scale: int = 1,
                 max_period: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(1, size, scale, max_period, rng)
