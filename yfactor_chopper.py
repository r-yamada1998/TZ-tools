#!/usr/bin/env python3
"""
Receiver noise temperature of the four IF bands with the rotating chopper.

Does not touch the SIS bias or the LO: set them beforehand (initialize.py,
or the values of your choice).

Usage (in the folder that holds this script):
    uv run python yfactor_chopper.py {filter,nofilter} FREQ

    FREQ is only a label for the record, e.g. 15.609 (the 1st LO SG frequency).

Sequence: for HU, HL, VU, VL, select the IF band, wait 10 s, take one 0-span
trace, close the path. Trx is computed from the hot and cold levels in each
trace (rx_performance_analysis.py).
"""

import argparse
import time

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("filter_state", choices=["filter", "nofilter"])
    p.add_argument("freq", help="1st LO SG frequency, recorded as a label")
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF band [s]")
    tm.add_rsky_arg(p)
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use={"sa", "switch"})
    if not args.keep_sa:
        dev.setup_sa("chopper")
    rec = tm.Recorder("yfactor_chopper", {
        **tm.base_meta(dev, args),
        "filter_state": args.filter_state,
        "freq": args.freq,
        "sa_params": dev.sa_params(),
        "note": "SIS bias and LO are whatever was set before this run.",
    }, columns=["band"])

    tm.ask_yes("start")
    for band in tm.BANDS:
        print(band, "is measuring...")
        dev.select_band(band)
        time.sleep(args.settle)
        rec.add(dev.fresh_trace(), band=band)
        dev.close_band(band)

    tm.finish(rec)


if __name__ == "__main__":
    main()
