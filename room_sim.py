from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple


@dataclass
class RoomGeometry:
    length_m: float = 7.0
    width_m: float = 7.0
    height_m: float = 4.0

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.height_m

    @property
    def wall_area_m2(self) -> float:
        wall1 = self.length_m * self.height_m
        wall2 = self.width_m * self.height_m
        return 2.0 * (wall1 + wall2)

    @property
    def roof_area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def envelope_area_m2(self) -> float:
        # Walls + roof. Floor is assumed to be near ground temperature for summer.
        return self.wall_area_m2 + self.roof_area_m2


@dataclass
class ThermalParams:
    air_density_kg_m3: float = 1.2
    air_cp_j_kgk: float = 1005.0
    ach_per_hour: float = 0.5

    # Typical summer envelope values (overall heat transfer coefficients).
    u_wall_w_m2k: float = 0.6
    u_roof_w_m2k: float = 0.3

    # Internal gains.
    people_count: int = 2
    sensible_w_per_person: float = 100.0
    equipment_w: float = 0.0
    solar_w: float = 0.0

    # AC cooling capacity and control.
    ac_capacity_w: float = 3500.0  # ~1 ton cooling
    ac_deadband_c: float = 0.5
    ac_kp_w_per_k: float = 800.0  # proportional cooling gain

    # Effective wall thermal mass for the 2-node model.
    wall_areal_heat_capacity_j_m2k: float = 150000.0
    h_in_w_m2k: float = 3.0


@dataclass
class SimulationState:
    time_s: float
    air_temp_c: float
    wall_temp_c: float | None = None


def _internal_gains_w(params: ThermalParams) -> float:
    people_w = params.people_count * params.sensible_w_per_person
    return people_w + params.equipment_w + params.solar_w


def _infiltration_w(
    params: ThermalParams,
    geom: RoomGeometry,
    t_out_c: float,
    t_air_c: float,
) -> float:
    flow_m3_s = params.ach_per_hour * geom.volume_m3 / 3600.0
    m_dot = params.air_density_kg_m3 * flow_m3_s
    return m_dot * params.air_cp_j_kgk * (t_out_c - t_air_c)


def _ac_cooling_w(
    params: ThermalParams,
    t_air_c: float,
    setpoint_c: float,
) -> float:
    if t_air_c <= setpoint_c + params.ac_deadband_c:
        return 0.0
    demand = params.ac_kp_w_per_k * (t_air_c - setpoint_c)
    return min(params.ac_capacity_w, max(0.0, demand))


class LumpedRoomModel:
    def __init__(self, geom: RoomGeometry, params: ThermalParams, dt_s: float = 300.0):
        self.geom = geom
        self.params = params
        self.dt_s = dt_s
        self._ua_w_k = (
            params.u_wall_w_m2k * geom.wall_area_m2
            + params.u_roof_w_m2k * geom.roof_area_m2
        )

        air_mass = params.air_density_kg_m3 * geom.volume_m3
        self._c_air_j_k = air_mass * params.air_cp_j_kgk

    def step(
        self,
        state: SimulationState,
        setpoint_c: float,
        t_out_c: float,
    ) -> Tuple[SimulationState, Dict[str, float]]:
        q_env = self._ua_w_k * (t_out_c - state.air_temp_c)
        q_inf = _infiltration_w(self.params, self.geom, t_out_c, state.air_temp_c)
        q_int = _internal_gains_w(self.params)
        q_ac = _ac_cooling_w(self.params, state.air_temp_c, setpoint_c)

        net_q = q_env + q_inf + q_int - q_ac
        d_t = (net_q / self._c_air_j_k) * self.dt_s
        next_air = state.air_temp_c + d_t

        next_state = SimulationState(
            time_s=state.time_s + self.dt_s,
            air_temp_c=next_air,
            wall_temp_c=None,
        )
        info = {
            "q_env_w": q_env,
            "q_inf_w": q_inf,
            "q_int_w": q_int,
            "q_ac_w": q_ac,
        }
        return next_state, info


class TwoNodeRoomModel:
    def __init__(self, geom: RoomGeometry, params: ThermalParams, dt_s: float = 300.0):
        self.geom = geom
        self.params = params
        self.dt_s = dt_s

        self._a_env_m2 = geom.envelope_area_m2
        self._r_in_k_w = 1.0 / (params.h_in_w_m2k * self._a_env_m2)
        self._r_out_k_w = 1.0 / (
            params.u_wall_w_m2k * geom.wall_area_m2
            + params.u_roof_w_m2k * geom.roof_area_m2
        )

        air_mass = params.air_density_kg_m3 * geom.volume_m3
        self._c_air_j_k = air_mass * params.air_cp_j_kgk
        self._c_wall_j_k = params.wall_areal_heat_capacity_j_m2k * self._a_env_m2

    def step(
        self,
        state: SimulationState,
        setpoint_c: float,
        t_out_c: float,
    ) -> Tuple[SimulationState, Dict[str, float]]:
        if state.wall_temp_c is None:
            wall_temp = state.air_temp_c
        else:
            wall_temp = state.wall_temp_c

        q_inf = _infiltration_w(self.params, self.geom, t_out_c, state.air_temp_c)
        q_int = _internal_gains_w(self.params)
        q_ac = _ac_cooling_w(self.params, state.air_temp_c, setpoint_c)

        q_air_to_wall = (wall_temp - state.air_temp_c) / self._r_in_k_w
        q_wall_to_out = (t_out_c - wall_temp) / self._r_out_k_w

        d_air = (q_air_to_wall + q_inf + q_int - q_ac) / self._c_air_j_k * self.dt_s
        d_wall = (q_wall_to_out - q_air_to_wall) / self._c_wall_j_k * self.dt_s

        next_air = state.air_temp_c + d_air
        next_wall = wall_temp + d_wall

        next_state = SimulationState(
            time_s=state.time_s + self.dt_s,
            air_temp_c=next_air,
            wall_temp_c=next_wall,
        )
        info = {
            "q_inf_w": q_inf,
            "q_int_w": q_int,
            "q_ac_w": q_ac,
            "q_air_to_wall_w": q_air_to_wall,
            "q_wall_to_out_w": q_wall_to_out,
        }
        return next_state, info


ModelName = Literal["lumped", "two_node"]


class RoomSimulator:
    def __init__(
        self,
        model: ModelName = "lumped",
        geom: RoomGeometry | None = None,
        params: ThermalParams | None = None,
        dt_s: float = 300.0,
    ):
        self.geom = geom or RoomGeometry()
        self.params = params or ThermalParams()
        self.dt_s = dt_s

        if model == "lumped":
            self.model = LumpedRoomModel(self.geom, self.params, dt_s=dt_s)
        elif model == "two_node":
            self.model = TwoNodeRoomModel(self.geom, self.params, dt_s=dt_s)
        else:
            raise ValueError(f"Unknown model: {model}")

    def reset(self, initial_temp_c: float, time_s: float = 0.0) -> SimulationState:
        return SimulationState(time_s=time_s, air_temp_c=initial_temp_c, wall_temp_c=initial_temp_c)

    def step(
        self,
        state: SimulationState,
        setpoint_c: float,
        t_out_c: float,
    ) -> Tuple[SimulationState, Dict[str, float]]:
        return self.model.step(state, setpoint_c, t_out_c)


if __name__ == "__main__":
    sim = RoomSimulator(model="two_node", dt_s=300.0)
    state = sim.reset(initial_temp_c=30.0)
    for _ in range(12):
        state, info = sim.step(state, setpoint_c=24.0, t_out_c=35.0)
        print(state, info)
