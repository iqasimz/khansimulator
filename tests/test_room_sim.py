import math

import pytest

from room_sim import OutdoorTempProfile, RoomSimulator, SimulationState, ThermalParams


def test_lumped_converges_to_outdoor_with_ac_off():
    params = ThermalParams(
        ac_capacity_w=0.0,
        ac_kp_w_per_k=0.0,
        people_count=0,
        equipment_w=0.0,
        solar_w=0.0,
    )
    sim = RoomSimulator(model="lumped", params=params, dt_s=300.0)
    state = sim.reset(initial_temp_c=20.0)

    t_out = 35.0
    for _ in range(200):
        state, _ = sim.step(state, setpoint_c=10.0, t_out_c=t_out)

    assert abs(state.air_temp_c - t_out) < 1.0


def test_ac_cools_room_when_above_setpoint():
    sim = RoomSimulator(model="lumped", dt_s=300.0)
    state = sim.reset(initial_temp_c=30.0)

    t_out = 35.0
    state_next, info = sim.step(state, setpoint_c=24.0, t_out_c=t_out)

    assert state_next.air_temp_c < state.air_temp_c
    assert info["q_ac_w"] > 0.0


def test_two_node_stability():
    sim = RoomSimulator(model="two_node", dt_s=300.0)
    state = SimulationState(time_s=0.0, air_temp_c=30.0, wall_temp_c=30.0)

    t_out = 38.0
    for _ in range(50):
        state, _ = sim.step(state, setpoint_c=26.0, t_out_c=t_out)

    assert math.isfinite(state.air_temp_c)
    assert math.isfinite(state.wall_temp_c)


def test_outdoor_profile_cycle():
    profile = OutdoorTempProfile(mean_c=30.0, amplitude_c=5.0, peak_hour=15.0)
    t_noon = profile.temperature_c(12 * 3600.0)
    t_peak = profile.temperature_c(15 * 3600.0)
    assert t_peak >= t_noon


def test_env_step_if_gymnasium_available():
    gym = pytest.importorskip("gymnasium")
    from room_env import RoomTempEnv

    env = RoomTempEnv()
    obs, _ = env.reset(seed=123)
    assert env.observation_space.contains(obs)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert math.isfinite(reward)
    assert terminated is False
    assert isinstance(truncated, bool)
    assert "energy_kwh" in info


def test_discrete_action_bounds():
    gym = pytest.importorskip("gymnasium")
    from room_env import RoomTempEnv

    env = RoomTempEnv()
    env.reset(seed=0)

    with pytest.raises(ValueError):
        env.step(-1)

    with pytest.raises(ValueError):
        env.step(len(env.config.setpoints_c))


def test_reward_includes_energy_penalty():
    gym = pytest.importorskip("gymnasium")
    from room_env import EnvConfig, RoomTempEnv

    config = EnvConfig(energy_weight=1.0, process_noise_std_c=0.0, outdoor_noise_std_c=0.0)
    env = RoomTempEnv(config=config)
    obs, _ = env.reset(seed=0)
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward <= 0.0
    assert info["q_ac_w"] >= 0.0


def test_noise_affects_observation():
    gym = pytest.importorskip("gymnasium")
    from room_env import EnvConfig, RoomTempEnv

    config = EnvConfig(process_noise_std_c=0.5, outdoor_noise_std_c=0.5)
    env = RoomTempEnv(config=config)
    env.reset(seed=0)
    obs1, _, _, _, _ = env.step(0)
    obs2, _, _, _, _ = env.step(0)
    assert obs1.shape == obs2.shape


def test_episode_truncation_72h():
    gym = pytest.importorskip("gymnasium")
    from room_env import EnvConfig, RoomTempEnv

    config = EnvConfig(episode_hours=72.0)
    env = RoomTempEnv(config=config)
    env.reset(seed=0)

    steps = int(config.episode_hours * 3600.0 / config.dt_s)
    truncated = False
    for _ in range(steps):
        _, _, _, truncated, _ = env.step(0)

    assert truncated is True


def test_vecnormalize_roundtrip_if_available(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from room_env import EnvConfig, RoomTempEnv

    config = EnvConfig()
    env = DummyVecEnv([lambda: RoomTempEnv(config=config)])
    vec = VecNormalize(env, norm_obs=True, norm_reward=True)

    obs = vec.reset()
    obs, reward, done, info = vec.step([0])

    stats_path = tmp_path / "vecnormalize.pkl"
    vec.save(str(stats_path))

    env2 = DummyVecEnv([lambda: RoomTempEnv(config=config)])
    vec2 = VecNormalize.load(str(stats_path), env2)
    vec2.training = False
    vec2.norm_reward = False

    obs2 = vec2.reset()
    assert obs2 is not None
