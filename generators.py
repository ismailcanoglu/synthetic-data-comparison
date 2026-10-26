"""Controlled data generators v2. Pilot only; not a frozen benchmark.

S1: overlapping imbalanced class-conditionals.
S2a/S2b: nonlinear concentric rings with different radial noise.
S3a: S1 plus test-time nuisance z2 shift.
S3b: S1 plus test-time informative z0 shift shared across classes.

Important: S3b changes P(X) and may also change P(Y|X); do NOT
claim strict covariate shift without verifying conditional invariance.
"""
from dataclasses import dataclass
import numpy as np

SCENARIOS = ('s1', 's2a', 's2b', 's3a', 's3b')

@dataclass(frozen=True)
class Settings:
    minority_probability: float = 0.10
    observation_noise: float = 0.08
    ring_noise_easy: float = 0.28
    ring_noise_moderate: float = 0.85  # pilot candidate; not tuned against SMOTE/CTGAN
    shift_magnitude: float = 1.5


def sample(n: int, seed: int, scenario: str, split: str = 'train', settings: Settings = Settings()):
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if n < 2:
        raise ValueError('n must be >=2')
    # Independent deterministic streams for labels, latents, geometry, measurement noise.
    # split-specific key makes train/test independent even for the same seed.
    split_id = 0 if split == 'train' else 1
    ss = np.random.SeedSequence([int(seed), split_id])
    ry, rz, rg, re = [np.random.default_rng(s) for s in ss.spawn(4)]
    y = ry.binomial(1, settings.minority_probability, size=n).astype(int)
    z = rz.normal(size=(n, 3))
    if scenario in ('s2a', 's2b'):
        noise = settings.ring_noise_easy if scenario == 's2a' else settings.ring_noise_moderate
        angle = rg.uniform(0, 2*np.pi, size=n)
        radius = np.where(y == 1, 2.3, .55) + rg.normal(0, noise, size=n)
        z[:, 0] = radius*np.cos(angle)
        z[:, 1] = radius*np.sin(angle)
        z[:, 2] += .4*y
    else:
        z[:, 0] += 1.1*y
        z[:, 1] -= .8*y
        if split == 'test' and scenario == 's3a':
            z[:, 2] += settings.shift_magnitude
        if split == 'test' and scenario == 's3b':
            z[:, 0] += settings.shift_magnitude
    x = np.column_stack([z, z[:,0]*z[:,1], z[:,0]**2, np.sin(z[:,2])])
    x += re.normal(0, settings.observation_noise, size=x.shape)
    return x, y
