#!/usr/bin/env python3
"""
Receiver noise temperature of the four IF bands, HOT and COLD switched by
hand with the slider (OLD MMC switch to Hot, then Cold).

Does not touch the SIS bias or the LO: set them beforehand.

Usage (in the folder that holds this script):
    uv run python yfactor_hotcold.py {filter,nofilter} [--freq FREQ]

Sequence: with the HOT load in place, for HU, HL, VU, VL select the IF band,
wait 10 s, take one trace, close the path. Then the same with the COLD load.
"y" is asked only when the load is to be changed (--ask-each-band: before
every band too). With averaging on (TZ manual: 20 sweeps), each trace is read
after the average has been restarted and completed.

Result: Trx of each band, and Trx versus IF frequency (4-8 GHz).
"""

import argparse
import time

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("filter_state", choices=["filter", "nofilter"])
    p.add_argument("--freq", help="1st LO SG frequency, recorded as a label (asked if omitted)")
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF band [s]")
    p.add_argument("--ask-each-band", action="store_true",
                   help="wait for y before every band")
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use={"sa", "switch"})
    freq = args.freq if args.freq is not None else input("freq = ")
    if not args.keep_sa:
        dev.setup_sa("slider")
    rec = tm.Recorder("yfactor_hotcold", {
        **tm.base_meta(dev, args),
        "filter_state": args.filter_state,
        "freq": freq,
        "sa_params": dev.sa_params(),
        "note": "SIS bias and LO are whatever was set before this run.",
    }, columns=["load", "band"])

    for load in ["hot", "cold"]:
        tm.ask_yes(f"Set the OLD MMC switch to {load.upper()}, then start")
        tm.set_load(load)
        for band in tm.BANDS:
            dev.select_band(band)
            print(band, "is setting...")
            time.sleep(args.settle)
            if args.ask_each_band:
                tm.ask_yes("ready")
            rec.add(dev.fresh_trace(), load=load, band=band)
            dev.close_band(band)

    tm.finish(rec)


if __name__ == "__main__":
    main()
