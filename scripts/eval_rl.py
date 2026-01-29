from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from room_env import EnvConfig, RoomTempEnv

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover - handled at runtime by users
    raise ImportError(
        "stable-baselines3 is required for evaluation. Install with "
        "`pip install stable-baselines3`."
    ) from exc


def make_env(seed: int, config: EnvConfig):
    def _thunk():
        env = RoomTempEnv(config=config)
        env.reset(seed=seed)
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO or SAC on the room temp env.")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved model.")
    parser.add_argument("--vecnorm", type=Path, default=None, help="Path to VecNormalize stats.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-hours", type=float, default=72.0)
    parser.add_argument("--outdoor-noise", type=float, default=0.5)
    parser.add_argument("--process-noise", type=float, default=0.1)
    args = parser.parse_args()

    config = EnvConfig(
        episode_hours=args.episode_hours,
        outdoor_noise_std_c=args.outdoor_noise,
        process_noise_std_c=args.process_noise,
    )

    env = DummyVecEnv([make_env(args.seed, config)])
    if args.vecnorm:
        env = VecNormalize.load(str(args.vecnorm), env)
        env.training = False
        env.norm_reward = False

    algo_cls = PPO if args.algo == "ppo" else SAC
    model = algo_cls.load(str(args.model), env=env)

    returns = []
    comfort = []
    energy = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_comfort = 0.0
        ep_energy = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward[0])
            info0 = info[0]
            ep_energy = float(info0.get("energy_kwh", ep_energy))
            ep_comfort += abs(float(info0.get("air_temp_c", 0.0)) - float(info0.get("setpoint_c", 0.0)))
            steps += 1

        returns.append(ep_return)
        comfort.append(ep_comfort / max(steps, 1))
        energy.append(ep_energy)

    print(f"Episodes: {args.episodes}")
    print(f"Return mean/std: {np.mean(returns):.3f} / {np.std(returns):.3f}")
    print(f"Comfort mean (avg |T-setpoint|): {np.mean(comfort):.3f} C")
    print(f"Energy mean (kWh): {np.mean(energy):.3f}")


if __name__ == "__main__":
    main()
