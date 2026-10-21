#!/usr/bin/env python3
"""
Set the LO attenuator currents, as the ROS script set_loatt_level_direction-ver2.py did.

Usage (in the folder that holds this script):
    uv run python set_loatt.py H V        [mA], e.g. 23 23 (ROS operating values)

Ports follow measurement.loatt_port in device_config_ros.toml (H = GPIB 4,
V = GPIB 5). The values stay after the script ends.
"""

import argparse

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("h", type=float, help="H current [mA]")
    p.add_argument("v", type=float, help="V current [mA]")
    tm.add_common_args(p)
    args = p.parse_args()

    for value in (args.h, args.v):
        if not 0.0 <= value <= 40.0:
            raise SystemExit(f"Aborted: {value} mA is outside 0-40 mA.")
    rig = tm.start(args, use={"loatt"})
    rig.set_loatt("H", args.h)
    rig.set_loatt("V", args.v)
    ports = rig.m["loatt_port"]
    print(f"LO attenuator set: H (port {ports['H']}) = {args.h} mA, V (port {ports['V']}) = {args.v} mA")


if __name__ == "__main__":
    main()
