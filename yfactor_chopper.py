#!/usr/bin/env python3
"""
Receiver noise temperature of the four IF bands with the rotating chopper.

Port of the ROS script yfactor_chopper_directon.py. Does not touch the SIS bias
or the LO: set them beforehand (initialize.py, or the values of your choice).

Usage:
    cd ~/tz_workspace && uv run python yfactor_chopper.py {filter,nofilter} FREQ

    FREQ is only a label for the record, e.g. 15.609 (the 1st LO SG frequency).

Sequence (as in ROS): for HU, HL, VU, VL, open the IF switch path, wait 10 s,
take one 0-span trace, close the path. Trx is computed from the hot and cold
levels in each trace (tz_analysis.py).
"""

import argparse

import tz_measure as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("filter_state", choices=["filter", "nofilter"])
    p.add_argument("freq", help="1st LO SG frequency, recorded as a label")
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF path [s] (ROS: 10)")
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sa", "switch"})
    if not args.keep_sa:
        rig.setup_sa("sa_chopper")
    rec = tm.Recorder("yfactor_chopper", {
        **tm.base_meta(rig, args),
        "filter_state": args.filter_state,
        "freq": args.freq,
        "sa_params": rig.sa_params(),
        "note": "SIS bias and LO are whatever was set before this run.",
    }, columns=["band"])

    tm.ask_yes("start")
    for band in tm.BANDS:
        print(band, "is measuring...")
        rig.open_band(band)
        tm.sleep(rig, args.settle)
        rec.add(rig.trace(), band=band)
        rig.close_band(band)

    tm.finish(rec)


if __name__ == "__main__":
    main()
