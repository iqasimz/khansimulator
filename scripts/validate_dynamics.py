from __future__ import annotations

import argparse

from room_sim import OutdoorTempProfile, RoomSimulator


def run_sim(
    model: str,
    initial_temp_c: float,
    setpoint_c: float,
    hours: float,
    dt_s: float,
    use_profile: bool,
):
    sim = RoomSimulator(model=model, dt_s=dt_s)
    state = sim.reset(initial_temp_c=initial_temp_c)
    profile = OutdoorTempProfile()
    steps = int(hours * 3600.0 / dt_s)

    for step in range(steps):
        t_out = profile.temperature_c(state.time_s) if use_profile else 35.0
        state, info = sim.step(state, setpoint_c=setpoint_c, t_out_c=t_out)
        hour = state.time_s / 3600.0
        wall_temp = state.wall_temp_c if state.wall_temp_c is not None else float("nan")
        print(f"{hour:.2f},{state.air_temp_c:.3f},{wall_temp:.3f},{t_out:.3f},{info.get('q_ac_w', 0.0):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Validate room thermal dynamics.")
    parser.add_argument("--model", choices=["lumped", "two_node"], default="two_node")
    parser.add_argument("--initial", type=float, default=30.0, help="Initial indoor temp (C).")
    parser.add_argument("--setpoint", type=float, default=24.0, help="Target setpoint (C).")
    parser.add_argument("--hours", type=float, default=6.0, help="Simulation duration (hours).")
    parser.add_argument("--dt", type=float, default=300.0, help="Time step (s).")
    parser.add_argument(
        "--outdoor-profile",
        action="store_true",
        help="Use sinusoidal outdoor temperature profile instead of fixed 35C.",
    )

    args = parser.parse_args()
    run_sim(
        model=args.model,
        initial_temp_c=args.initial,
        setpoint_c=args.setpoint,
        hours=args.hours,
        dt_s=args.dt,
        use_profile=args.outdoor_profile,
    )


if __name__ == "__main__":
    main()
