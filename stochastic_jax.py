from typing import Tuple
import numpy as np
import jax
from jax import numpy as jnp
import distrax
from flax import linen as nn

from noise_process import ColoredNoiseProcess


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state):
        trunk = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
        ])
        log_std_head = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        mu_head = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        hidden = trunk(state)
        mu, log_std = mu_head(hidden), log_std_head(hidden)
        log_std = jnp.clip(log_std, -20, 2)

        dist = TanhNormal(mu, jnp.exp(log_std))
        return dist


class ColoredActor(nn.Module):
    beta: float
    seq_len: int
    batch_size: int
    action_dim: int
    hidden_dim: int
    max_action: float = 1.0
    rng: np.random.Generator

    def setup(self):
        self.trunk = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
        ])
        self.log_std_head = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        self.mu_head = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        self.noise = ColoredNoiseProcess(beta=self.beta,
                                         size=(self.batch_size, self.action_dim, self.seq_len),
                                         rng=self.rng)

    def __call__(self, state: jax.Array) -> Tuple[jax.Array, jax.Array]:
        hidden = self.trunk(state)
        mu, log_std = self.mu_head(hidden), self.log_std_head(hidden)
        log_std = jnp.clip(log_std, -20, 2)

        eps = jnp.asarray(self.noise.sample())
        action = mu + eps * jnp.exp(log_std)
        tanh_action = jnp.tanh(action)

        policy_distribution = distrax.Normal(mu, jnp.exp(log_std))
        log_prob = policy_distribution.log_prob(action).sum(-1)
        log_prob = log_prob - jnp.log(1 - jnp.square(tanh_action) + 1e-6).sum(-1)

        return tanh_action * self.max_action, log_prob
