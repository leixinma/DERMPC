"""Simulation environment: state management, disturbances, history recording."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .dynamics import DERBeam2D


class Simulator:
    """Wraps DERBeam2D with state management and disturbance generation."""

    def __init__(self, beam: DERBeam2D, cfg: dict):
        self.beam = beam
        self.dt: float = cfg["sim"]["dt"]
        self.pbd_iter: int = cfg["sim"].get("pbd_iter", 3)
        self.obs_noise: float = cfg["sim"].get("obs_noise_std", 0.0)
        self.dist_cfg = cfg.get("disturbance", {})

        self.x: np.ndarray = beam.rest_state()
        self.t: float = 0.0
        self.step_count: int = 0
        self.history: Dict[str, list] = {
            "x": [], "u": [], "t": [], "ke": [], "tip_y": [],
        }

    def reset(self, x0: np.ndarray | None = None):
        self.x = x0.copy() if x0 is not None else self.beam.rest_state()
        self.t = 0.0
        self.step_count = 0
        self.history = {k: [] for k in self.history}

    def step(self, u: np.ndarray | None = None) -> np.ndarray:
        if u is None:
            u = np.zeros(self.beam.n_u)

        f_ext = self._disturbance(self.t)
        self.x = self.beam.step(self.x, u, self.dt, f_ext, self.pbd_iter)
        self.t += self.dt
        self.step_count += 1

        self.history["x"].append(self.x.copy())
        self.history["u"].append(u.copy())
        self.history["t"].append(self.t)
        self.history["ke"].append(self.beam.kinetic_energy(self.x))
        pos = self.beam.get_positions(self.x)
        self.history["tip_y"].append(pos[-1, 1])

        return self.x

    def observe(self) -> np.ndarray:
        if self.obs_noise > 0:
            nq = self.beam.n_free * 2
            noise = np.zeros(self.beam.nx)
            noise[:nq] = np.random.normal(0, self.obs_noise, nq)
            return self.x + noise
        return self.x.copy()

    def _disturbance(self, t: float) -> np.ndarray:
        f = np.zeros(self.beam.n_free * 2)
        if not self.dist_cfg:
            return f

        gnode = self.dist_cfg.get("node", self.beam.n_nodes - 1)
        fi = gnode - self.beam.n_fixed
        d = np.array(self.dist_cfg.get("direction", [0, 1]), dtype=float)
        d = d / np.linalg.norm(d)
        dtype = self.dist_cfg.get("type", "sinusoidal")

        if dtype == "sinusoidal":
            amp = self.dist_cfg.get("amplitude", 1.0)
            freq = self.dist_cfg.get("frequency", 1.0)
            F = amp * np.sin(2 * np.pi * freq * t)
        elif dtype == "step":
            amp = self.dist_cfg.get("amplitude", 1.0)
            F = amp
        else:
            F = 0.0

        f[2 * fi: 2 * fi + 2] = F * d
        return f

    def get_history_arrays(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self.history.items()}
