#!/usr/bin/env python3
"""
Receiver noise temperature of the four IF bands with HOT and COLD loads
placed by hand.

Port of the ROS script yfactor_sa_direction.py. Does not touch the SIS bias or
the LO: set them beforehand.

Usage:
    cd ~/tz_workspace && uv run python yfactor_hotcold.py {filter,nofilter} [--freq FREQ]

Sequence (as in ROS): with the HOT load in place, for HU, HL, VU, VL open the
IF switch path, wait 10 s, wait for "y", take one trace, close the path. Then
the same with the COLD load.
"""

import argparse

import tz_measure as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("filter_state", choices=["filter", "nofilter"])
    p.add_argument("--freq", help="1st LO SG frequency, recorded as a label (asked if omitted)")
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF path [s] (ROS: 10)")
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sa", "switch"})
    freq = args.freq if args.freq is not None else input("freq = ")
    if not args.keep_sa:
        rig.setup_sa("sa_chopper")
    rec = tm.Recorder("yfactor_hotcold", {
        **tm.base_meta(rig, args),
        "filter_state": args.filter_state,
        "freq": freq,
        "sa_params": rig.sa_params(),
        "note": "SIS bias and LO are whatever was set before this run.",
    }, columns=["load", "band"])

    for load in ["hot", "cold"]:
        tm.ask_yes(f"{load.upper()} start")
        rig.load = load
        for band in tm.BANDS:
            rig.open_band(band)
            print(band, "is setting...")
            tm.sleep(rig, args.settle)
            tm.ask_yes("ready")
            rec.add(rig.trace(), load=load, band=band)
            rig.close_band(band)

    tm.finish(rec)


if __name__ == "__main__":
    main()
