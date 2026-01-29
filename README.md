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
  - `LumpedRoomModel` and `TwoNodeRoomModel`.
  - `RoomSimulator` wrapper.

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
