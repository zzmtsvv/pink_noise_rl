# modules to work with deterministic actors (e.g. for TD3 like)
from typing import Any, Optional
import numpy as np
from numpy.random._generator import Generator as Generator
from numpy.typing import ArrayLike
from noise_process import ColoredNoiseProcess


class ColoredActionNoise:
    def __init__(self,
                 beta: ArrayLike,
                 sigma: ArrayLike,
                 seq_len: int,
                 action_dim: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        assert (action_dim is not None) == np.isscalar(beta), "mazafuka wrong dtypes"

        if np.isscalar(sigma):
            self.sigma = np.full(action_dim or len(beta))
        else:
            self.sigma = np.asarray(sigma)
        
        if np.isscalar(beta):
            self.beta = beta
            self.process = ColoredNoiseProcess(beta=self.beta,
                                               scale=self.sigma,
                                               size=(action_dim, seq_len),
                                               rng=rng)
        else:
            self.beta = np.asarray(beta)
            self.process = [
                ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                for b, s in zip(self.beta, self.sigma)
            ]
        
        self.is_beta_scalar = np.isscalar(self.beta)
        
    def __call__(self) -> np.ndarray:
        if self.is_beta_scalar:
            return self.process.sample()
        
        return np.asarray([p.sample() for p in self.process])


class PinkActionNoise(ColoredActionNoise):
    def __init__(self,
                 sigma: ArrayLike,
                 seq_len: int,
                 action_dim: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(1, sigma, seq_len, action_dim, rng)
