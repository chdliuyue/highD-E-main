"""Simple Intelligent Driver Model (IDM) implementation.

This module provides a minimal IDM class used to generate ghost car
trajectories as smooth baselines for conflict episodes.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


class IDM:
    """Intelligent Driver Model for longitudinal motion simulation.

    Parameters are kept simple and should ideally be calibrated using
    baseline episodes in the dataset.
    """

    def __init__(self, v0: float, T: float, s0: float, a_max: float, b_comf: float, delta: float = 4.0) -> None:
        """
        Initialize the IDM with standard parameters.

        Args:
            v0: Desired speed (m/s).
            T: Desired time headway (s).
            s0: Minimum gap (m).
            a_max: Maximum acceleration (m/s^2).
            b_comf: Comfortable deceleration (m/s^2).
            delta: Acceleration exponent, typically 4.
        """
        self.v0 = v0
        self.T = T
        self.s0 = s0
        self.a_max = a_max
        self.b_comf = b_comf
        self.delta = delta

    def _desired_gap(self, v: float, dv: float) -> float:
        """Compute dynamic desired gap s*.

        Args:
            v: Subject vehicle speed.
            dv: Speed difference (v - v_leader).

        Returns:
            Desired dynamic gap.
        """
        return self.s0 + max(0.0, v * self.T + (v * dv) / (2.0 * np.sqrt(self.a_max * self.b_comf)))

    def simulate(
        self,
        t: np.ndarray,
        leader_pos: np.ndarray,
        leader_speed: np.ndarray,
        s0_init: float,
        v_init: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the ghost car longitudinal state using forward Euler.

        Args:
            t: Time grid (seconds).
            leader_pos: Leader position array aligned with ``t``.
            leader_speed: Leader speed array aligned with ``t``.
            s0_init: Initial gap to the leader at ``t[0]``.
            v_init: Initial speed of the ghost car at ``t[0]``.

        Returns:
            Tuple of (ghost_pos, ghost_speed), each matching the shape of ``t``.
        """
        if len(t) == 0:
            return np.array([]), np.array([])

        t = np.asarray(t, dtype=float)
        leader_pos = np.asarray(leader_pos, dtype=float)
        leader_speed = np.asarray(leader_speed, dtype=float)

        dt_array = np.diff(t)
        dt = float(np.mean(dt_array)) if len(dt_array) > 0 else 0.0

        ghost_pos = np.zeros_like(t, dtype=float)
        ghost_speed = np.zeros_like(t, dtype=float)
        ghost_pos[0] = float(leader_pos[0] - s0_init)
        ghost_speed[0] = float(v_init)

        for i in range(len(t) - 1):
            dt_step = float(t[i + 1] - t[i]) if len(t) > 1 else dt
            gap = max(leader_pos[i] - ghost_pos[i], 1e-3)
            dv = ghost_speed[i] - leader_speed[i]
            desired_gap = self._desired_gap(ghost_speed[i], dv)
            acc = self.a_max * (1 - (ghost_speed[i] / max(self.v0, 1e-3)) ** self.delta - (desired_gap / gap) ** 2)
            acc = float(np.clip(acc, -self.b_comf, self.a_max))

            ghost_speed[i + 1] = max(0.0, ghost_speed[i] + acc * dt_step)
            ghost_pos[i + 1] = ghost_pos[i] + ghost_speed[i] * dt_step

        return ghost_pos, ghost_speed
