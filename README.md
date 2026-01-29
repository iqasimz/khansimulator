# Room Thermodynamics Simulator (Backend)

This project provides a backend-only thermodynamics simulator for a single room
environment intended for RL training. It models a 7 m x 7 m x 4 m room with
outdoor temperature influence, internal heat gains, and an air conditioner that
tracks a target setpoint. Two model variants are included:

- Lumped 1-zone model (fast, common for RL).
- Two-node RC model (air + wall thermal mass).

The simulator advances in discrete time steps (default 5 minutes) and does not
include any visualization.

## Geometry and Assumptions

- Room dimensions: 7 m x 7 m x 4 m.
- Air volume: 196 m^3.
- Envelope area: walls + roof (floor is omitted in summer assumption).
- Summer season assumption: outdoor temperature is higher than indoor.
- Standard envelope values (typical residential/commercial insulation):
  - Walls: U = 0.6 W/m^2-K
  - Roof: U = 0.3 W/m^2-K
- Infiltration: 0.5 air changes per hour (ACH).
- Internal gains: 2 people at 100 W sensible each.
- AC capacity: 3.5 kW (about 1 ton cooling).
- AC control: proportional cooling power with a 0.5 C deadband around setpoint.

All parameters are configurable in `ThermalParams`.

## Governing Equations

### Lumped 1-Zone Model (Air Only)

Energy balance on room air:

```
C_air * dT_air/dt = Q_env + Q_inf + Q_int - Q_ac

Q_env = UA * (T_out - T_air)
Q_inf = m_dot * c_p * (T_out - T_air)
Q_int = Q_people + Q_equipment + Q_solar
Q_ac  = cooling power from controller
```

Where:
- `C_air = m_air * c_p` is the thermal capacitance of indoor air.
- `UA` is the total envelope heat transfer coefficient (walls + roof).
- `m_dot` is the infiltration mass flow from ACH.

### Two-Node RC Model (Air + Walls)

Two coupled nodes for air and walls:

```
C_air  * dT_air/dt  = (T_wall - T_air)/R_in + Q_inf + Q_int - Q_ac
C_wall * dT_wall/dt = (T_out  - T_wall)/R_out + (T_air - T_wall)/R_in
```

Where:
- `R_in` models convection between indoor air and the wall mass.
- `R_out` models conduction through the envelope to outdoors.
- `C_wall` is the wall thermal capacitance based on areal heat capacity.

This model captures thermal inertia and slower recovery when the AC turns off.

## AC Control Law

The AC uses a proportional controller with saturation:

```
if T_air <= setpoint + deadband:
    Q_ac = 0
else:
    Q_ac = min(Q_max, K_p * (T_air - setpoint))
```

This mimics typical cooling behavior: stronger cooling when the room is far
above the target, zero cooling inside the deadband.

## Time Step

- Default time step: 300 seconds (5 minutes).
- The simulator integrates forward with an explicit Euler step.

## Code Layout

- `room_sim.py` contains:
  - `RoomGeometry` (room dimensions and derived areas/volume).
  - `ThermalParams` (all physical parameters and gains).
  - `SimulationState` (air temperature, wall temperature, time).
  - `OutdoorTempProfile` (daily sinusoidal summer profile).
  - `LumpedRoomModel` and `TwoNodeRoomModel`.
  - `RoomSimulator` wrapper.
- `room_env.py` contains:
  - `RoomTempEnv` (Gymnasium-compatible environment).
  - `EnvConfig` (observation/action configuration).

## Usage

Example loop (two-node model):

```python
from room_sim import RoomSimulator

sim = RoomSimulator(model="two_node", dt_s=300.0)
state = sim.reset(initial_temp_c=30.0)

for _ in range(12):
    state, info = sim.step(state, setpoint_c=24.0, t_out_c=35.0)
    print(state, info)
```

For RL training, you typically treat `setpoint_c` as the action and
`air_temp_c` (plus optional `wall_temp_c`) as the state.

Gymnasium environment usage:

```python
from room_env import RoomTempEnv

env = RoomTempEnv()
obs, _ = env.reset(seed=123)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Parameter Tuning Notes

- For tighter envelopes, reduce `u_wall_w_m2k` and `u_roof_w_m2k`.
- For older buildings, increase `ach_per_hour` (e.g., 1.0 to 2.0).
- For higher cooling power, increase `ac_capacity_w`.
- Wall thermal inertia can be adjusted by
  `wall_areal_heat_capacity_j_m2k`.

## Scope and Limitations

- Single-zone model (no spatial gradients).
- No humidity or latent loads.
- No solar geometry; solar heat gains are a fixed parameter.
- Floor conduction ignored for summer assumptions.
- No detailed HVAC dynamics (coil, fan, or compressor cycles).

These choices keep the simulator fast and stable for RL.

## Tests

The test suite validates core physics behavior and environment wiring:

```
pytest -q
```

## Validation Script

The script below prints CSV rows for plotting cool‑down or warm‑up curves.
Columns: `hour,air_temp_c,wall_temp_c,outdoor_temp_c,q_ac_w`.

```
python scripts/validate_dynamics.py --model two_node --initial 30 --setpoint 24 --hours 6
```

Use a sinusoidal outdoor profile instead of fixed 35 C:

```
python scripts/validate_dynamics.py --outdoor-profile
```

Plot the CSV data:

```
python scripts/validate_dynamics.py --outdoor-profile > /tmp/dynamics.csv
python scripts/plot_dynamics.py /tmp/dynamics.csv --out /tmp/dynamics.png
```

## Training (PPO or SAC)

Training script (Stable-Baselines3, with VecNormalize enabled by default):

```
python scripts/train_rl.py --algo ppo --timesteps 500000
python scripts/train_rl.py --algo sac --timesteps 500000
```

Optional flags:

- `--no-vecnorm` to disable observation/reward normalization.
- `--logdir` to set output directory (default `runs/room_rl`).
- `--seed` for reproducibility.
- `--episode-hours` to set the time limit (default 72).
- `--outdoor-noise` and `--process-noise` to control stochasticity.

## Evaluation

Evaluate a trained model over multiple episodes:

```
python scripts/eval_rl.py --algo ppo --model runs/room_rl/ppo_room_model.zip --vecnorm runs/room_rl/vecnormalize.pkl
```

## Requirements

```
pip install -r requirements.txt
```

## End-to-End Run

End-to-end pipeline (validate → plot → train → evaluate):

```
bash scripts/run_end_to_end.sh
```

What it does:

1) Runs a short validation simulation and saves CSV + plot.
   - CSV: `CSV_PATH` (default `/tmp/dynamics.csv`)
   - Plot: `PLOT_PATH` (default `/tmp/dynamics.png`)
2) Trains PPO or SAC with normalization enabled by default.
   - Model: `${LOGDIR}/${ALGO}_room_model.zip`
   - VecNormalize stats: `${LOGDIR}/vecnormalize.pkl`
3) Evaluates the trained model and prints summary stats.

Configure via environment variables:

```
ALGO=sac TIMESTEPS=300000 LOGDIR=runs/room_sac EPISODES=10 \
OUTDOOR_NOISE=0.7 PROCESS_NOISE=0.2 EPISODE_HOURS=72 \
VALIDATE_HOURS=8 VALIDATE_SETPOINT=24 VALIDATE_INITIAL=30 \
CSV_PATH=/tmp/dynamics.csv PLOT_PATH=/tmp/dynamics.png \
bash scripts/run_end_to_end.sh
```

## Suggested Next Steps

1) Calibrate parameters
   - If you have HVAC specs, set `ac_capacity_w` and `ac_kp_w_per_k`.
   - If you know insulation quality, tune `u_wall_w_m2k`, `u_roof_w_m2k`, and `ach_per_hour`.

2) Validate dynamics
   - Check cool-down and warm-up curves visually.
   - Compare lumped vs two-node behavior before training.

3) Add evaluation
   - Run trained policies and log comfort/energy tradeoffs.
   - Plot trajectories and compute average reward per episode.

## Reward / Cost Function (General)

The environment returns a generic reward that balances comfort and energy:

```
reward = - ( |T_air - setpoint| + energy_weight * (Q_ac / Q_ac_max) )
```

- `energy_weight` scales energy cost relative to comfort.
- `Q_ac` is the instantaneous AC power draw.
- `Q_ac_max` is the max AC capacity.

You can replace this with a squared comfort penalty, banded constraints,
or a weighted multi-objective reward.

## Outdoor Temperature Profile

The default outdoor temperature uses a sinusoidal daily cycle with additive
Gaussian noise:

```
T_out(t) = mean + amplitude * cos(2π * (t_hour - peak_hour) / 24) + noise
```

Noise is controlled by `outdoor_noise_std_c` in `EnvConfig`.
Stochastic process noise is controlled by `process_noise_std_c` in `EnvConfig`.

## Action Space (Discrete)

The environment uses a discrete set of setpoints:

- Default setpoints: `[20, 22, 24, 26, 28]` C
- Action is the index into this list.
- Override via `EnvConfig.setpoints_c`.

## Progress Notes

- Implemented lumped and two-node RC thermal models.
- Added Gymnasium environment with discrete setpoint control.
- Added sinusoidal outdoor profile with noise.
- Added stochastic process noise to indoor temperatures.
- Set the default episode length to 72 hours.
- Added training script with normalization (VecNormalize).
- Added reward with comfort + energy penalty and energy tracking in `info`.
- Added tests validating physics behavior and env wiring.