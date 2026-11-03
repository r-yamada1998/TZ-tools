#!/usr/bin/env python3
"""
Receiver noise temperature versus LO Att. current, HL and VL bands.

The chopper must be rotating.

Usage (in the folder that holds this script):
    uv run python loatt_search.py START STOP NUM FREQ \
        [--bias V1 H1 H2 V2]

    START STOP NUM  LO Att. currents [mA], numpy.linspace(START, STOP, NUM)
    FREQ            1st LO SG frequency, recorded as a label
    --bias          SIS biases to hold during the scan (default: tuned values)

Sequence: set the SIS biases, wait for "y"; for each current set H, wait
20 s, set V; for HL and VL select the IF band, wait 10 s and take traces.
At the end the operator types the H and V currents to keep.

The V attenuator is read about 20 s after it is set (the two IF waits);
on 2026-10-26 runs up and down 20-23 mA showed no dependence on the
direction, only a drift of a few percent over minutes.
"""

import argparse
import time

import numpy as np

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("start", type=float)
    p.add_argument("stop", type=float)
    p.add_argument("num", type=int)
    p.add_argument("freq")
    p.add_argument("--bias", nargs=4, type=float, metavar=("V1", "H1", "H2", "V2"))
    p.add_argument("--att-settle", type=float, default=20.0,
                   help="wait after setting the H attenuator [s]")
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF band [s]")
    p.add_argument("--traces", type=int, default=1, help="traces per point")
    tm.add_rsky_arg(p)
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use={"sis", "sa", "switch", "loatt"})
    bias = dict(zip(tm.BIAS_NAMES, args.bias)) if args.bias else tm.tuned_bias(dev.cfg)
    att_list = np.linspace(args.start, args.stop, args.num)
    if not args.keep_sa:
        dev.setup_sa("chopper")

    rec = tm.Recorder("loatt_search", {
        **tm.base_meta(dev, args),
        "freq": args.freq,
        "bias_mV": bias,
        "sa_params": dev.sa_params(),
    }, columns=["att_mA", "band", "repeat"])

    dev.set_bias(**bias)
    print("SIS bias:", bias)
    tm.ask_yes("ready?")

    for att in att_list:
        att = round(float(att), 4)
        print(f"now setting att. = {att}  (please wait {args.att_settle:.0f} s)")
        dev.set_loatt("H", att)
        time.sleep(args.att_settle)
        dev.set_loatt("V", att)
        for band in ["HL", "VL"]:
            dev.select_band(band)
            dev.close_band(band)
            print("current measurement band:", band)
            time.sleep(args.settle)
            for k in range(args.traces):
                rec.add(dev.fresh_trace(), att_mA=att, band=band, repeat=k)

    tm.finish(rec)

    print("finished att. param search... please set att.")
    att_h = 23.0 if args.yes else float(input("set H-pol. att [mA]: "))
    att_v = 23.0 if args.yes else float(input("set V-pol. att [mA]: "))
    # No wait: nothing is measured after this.
    dev.set_loatt("H", att_h)
    dev.set_loatt("V", att_v)
    print(f"LO attenuator set: H {att_h} mA, V {att_v} mA")


if __name__ == "__main__":
    main()
