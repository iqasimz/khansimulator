from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np

from room_sim import OutdoorTempProfile, RoomSimulator, SimulationState, ThermalParams

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - handled at runtime by users/tests
    raise ImportError(
        "gymnasium is required for RoomTempEnv. Install with `pip install gymnasium`."
    ) from exc


@dataclass
class EnvConfig:
    model: str = "two_node"
    dt_s: float = 300.0
    episode_hours: float = 72.0
    setpoints_c: List[float] = field(default_factory=lambda: [20.0, 22.0, 24.0, 26.0, 28.0])
    initial_temp_low_c: float = 28.0
    initial_temp_high_c: float = 32.0
    energy_weight: float = 0.1
    include_wall_temp: bool = True
    include_time_of_day: bool = True
    outdoor_noise_std_c: float = 0.5
    process_noise_std_c: float = 0.1


class RoomTempEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        thermal_params: ThermalParams | None = None,
        outdoor_profile: OutdoorTempProfile | None = None,
    ):
        self.config = config or EnvConfig()
        self.params = thermal_params or ThermalParams()
        self.profile = outdoor_profile or OutdoorTempProfile()

        self.sim = RoomSimulator(
            model=self.config.model,
            params=self.params,
            dt_s=self.config.dt_s,
        )

        self.max_steps = int(self.config.episode_hours * 3600.0 / self.config.dt_s)
        self.action_space = spaces.Discrete(len(self.config.setpoints_c))

        obs_low = [0.0, -20.0, min(self.config.setpoints_c)]
        obs_high = [60.0, 60.0, max(self.config.setpoints_c)]

        if self.config.include_wall_temp:
            obs_low.insert(1, -20.0)
            obs_high.insert(1, 60.0)

        if self.config.include_time_of_day:
            obs_low.append(0.0)
            obs_high.append(24.0)

        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.state: SimulationState | None = None
        self.setpoint_c = float(max(self.config.setpoints_c))
        self.current_step = 0
        self.energy_kwh = 0.0
        self._rng = np.random.default_rng()

    def _get_outdoor_temp(self) -> float:
        time_s = self.state.time_s if self.state else 0.0
        base = self.profile.temperature_c(time_s)
        noise = self._rng.normal(0.0, self.config.outdoor_noise_std_c)
        return base + noise

    def _obs(self, t_out_c: float) -> np.ndarray:
        if not self.state:
            raise RuntimeError("Environment must be reset before stepping.")

        obs = [self.state.air_temp_c]
        if self.config.include_wall_temp:
            obs.append(self.state.wall_temp_c if self.state.wall_temp_c is not None else self.state.air_temp_c)
        obs.append(t_out_c)
        obs.append(self.setpoint_c)
        if self.config.include_time_of_day:
            obs.append((self.state.time_s / 3600.0) % 24.0)
        return np.array(obs, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed)

        initial_temp = rng.uniform(
            self.config.initial_temp_low_c,
            self.config.initial_temp_high_c,
        )

        self.state = self.sim.reset(initial_temp_c=initial_temp, time_s=0.0)
        self.setpoint_c = float(max(self.config.setpoints_c))
        self.current_step = 0
        self.energy_kwh = 0.0

        t_out = self._get_outdoor_temp()
        return self._obs(t_out), {}

    def step(self, action: int):
        if not self.state:
            raise RuntimeError("Environment must be reset before stepping.")

        idx = int(action)
        if idx < 0 or idx >= len(self.config.setpoints_c):
            raise ValueError("Action index out of range for configured setpoints.")
        setpoint = float(self.config.setpoints_c[idx])
        self.setpoint_c = setpoint

        t_out = self._get_outdoor_temp()
        next_state, info = self.sim.step(self.state, setpoint_c=setpoint, t_out_c=t_out)
        noisy_air = next_state.air_temp_c + self._rng.normal(0.0, self.config.process_noise_std_c)
        noisy_wall = next_state.wall_temp_c
        if noisy_wall is not None:
            noisy_wall = noisy_wall + self._rng.normal(0.0, self.config.process_noise_std_c)
        self.state = SimulationState(
            time_s=next_state.time_s,
            air_temp_c=noisy_air,
            wall_temp_c=noisy_wall,
        )
        self.current_step += 1

        q_ac_w = float(info.get("q_ac_w", 0.0))
        self.energy_kwh += q_ac_w * self.config.dt_s / 3_600_000.0

        comfort_penalty = abs(self.state.air_temp_c - setpoint)
        energy_penalty = self.config.energy_weight * (q_ac_w / max(self.params.ac_capacity_w, 1.0))
        reward = -(comfort_penalty + energy_penalty)

        truncated = self.current_step >= self.max_steps
        terminated = False

        info_out = dict(info)
        info_out["energy_kwh"] = self.energy_kwh
        info_out["outdoor_temp_c"] = t_out
        info_out["air_temp_c"] = self.state.air_temp_c
        info_out["setpoint_c"] = self.setpoint_c

        return self._obs(t_out), reward, terminated, truncated, info_out
