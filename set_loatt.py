#!/usr/bin/env python3
"""
Set the LO Att. currents.

Usage (in the folder that holds this script):
    uv run python set_loatt.py H V        [mA], e.g. 27 20

A larger current attenuates more. After the work, 70 70 leaves the LO weak
while the 1st LO is switched (receiver developer). The values stay after the
script ends.
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
        # TZ manual 3.3: 0-100 mA
        if not 0.0 <= value <= 100.0:
            raise SystemExit(f"Aborted: {value} mA is outside 0-100 mA.")
    dev = tm.start(args, use={"loatt"})
    dev.set_loatt("H", args.h)
    dev.set_loatt("V", args.v)
    ports = dev.m["local_attenuator"]["port"]
    print(f"LO attenuator set: H (port {ports['H']}) = {args.h} mA, V (port {ports['V']}) = {args.v} mA")


if __name__ == "__main__":
    main()
