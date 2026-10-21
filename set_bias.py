#!/usr/bin/env python3
"""
Set the four SIS biases, as the ROS script set_sis_vol_direction.py did.

Usage (in the folder that holds this script):
    uv run python set_bias.py V1 H1 H2 V2          e.g. 6.4 7.3 7.3 6.3
    uv run python set_bias.py --tuned              the operating values in device_config_ros.toml

Names, DA terminals and conversion follow device_config_ros.toml (the ROS
writer). The values stay after the script ends. Limits: bias_min and bias_max
in that file (0-9 mV).
"""

import argparse

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("values", nargs="*", type=float, metavar="MV", help="V1 H1 H2 V2 [mV]")
    p.add_argument("--tuned", action="store_true", help="use sis_bias_setter.tuned")
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sis"})
    if args.tuned:
        bias = dict(rig.cfg["sis_bias_setter"]["tuned"])
    elif len(args.values) == 4:
        bias = dict(zip(tm.BIAS_NAMES, args.values))
    else:
        raise SystemExit("Give V1 H1 H2 V2, or --tuned.")
    rig.set_bias(**bias)
    ch = rig.cfg["sis_bias_setter"]["channel"]
    print("SIS bias set: " + ", ".join(f"{k} ({ch[k]}) = {v} mV" for k, v in bias.items()))


if __name__ == "__main__":
    main()
