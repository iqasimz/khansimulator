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
        "stable-baselines3 is required for training. Install with "
        "`pip install stable-baselines3`."
    ) from exc


def make_env(seed: int, config: EnvConfig):
    def _thunk():
        env = RoomTempEnv(config=config)
        env.reset(seed=seed)
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on the room temp env.")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=Path, default=Path("runs/room_rl"))
    parser.add_argument("--no-vecnorm", action="store_true")
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
    if not args.no_vecnorm:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    logdir = args.logdir
    logdir.mkdir(parents=True, exist_ok=True)

    algo_cls = PPO if args.algo == "ppo" else SAC
    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logdir),
    )

    model.learn(total_timesteps=args.timesteps)

    model_path = logdir / f"{args.algo}_room_model"
    model.save(model_path)

    if hasattr(env, "save"):
        env.save(str(logdir / "vecnormalize.pkl"))

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
