#!/usr/bin/env python3
"""
Map the receiver noise temperature of one IF band over two SIS bias voltages.

The chopper must be rotating.

Usage (in the folder that holds this script):
    uv run python trx_map.py {HU,HL,VU,VL} \
        --bias1 6.0 8.0 21 --bias2 6.0 8.0 21 [--auto]

    --bias1 START STOP NUM   sets v1 and h1 together
    --bias2 START STOP NUM   sets v2 and h2 together
    (both polarizations move at once; only the chosen band is read)

Sequence: for each bias1 value, set v1 and h1, wait for "y" (skipped with
--auto); then for each bias2 value set v2 and h2, wait 5.8 s and take one
0-span trace. All four biases are set to 0 at the end; set the operating
values again afterwards (initialize.py or set_bias.py --tuned).
"""

import argparse
import time

import numpy as np

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("band", choices=tm.BANDS)
    p.add_argument("--bias1", nargs=3, type=float, required=True, metavar=("START", "STOP", "NUM"))
    p.add_argument("--bias2", nargs=3, type=float, required=True, metavar=("START", "STOP", "NUM"))
    p.add_argument("--auto", action="store_true", help="do not wait for y at each bias1 step")
    p.add_argument("--settle", type=float, default=5.8,
                   help="wait after setting bias2 [s]")
    tm.add_rsky_arg(p)
    tm.add_common_args(p)
    args = p.parse_args()

    vol1 = np.linspace(args.bias1[0], args.bias1[1], int(args.bias1[2]))
    vol2 = np.linspace(args.bias2[0], args.bias2[1], int(args.bias2[2]))
    print(f"{len(vol1)} x {len(vol2)} = {len(vol1) * len(vol2)} points, "
          f"about {len(vol1) * len(vol2) * args.settle / 60:.0f} min")

    dev = tm.start(args, use={"sis", "sa", "switch"})
    lim = dev.m["bias"]
    for v in np.concatenate([vol1, vol2]):
        if not lim["min"] <= v <= lim["max"]:
            raise SystemExit(f"Aborted: {v} mV is outside the bias limits.")
    if not args.keep_sa:
        dev.setup_sa("chopper")

    ch = dev.cfg["sis_bias_setter"]["channel"]
    rec = tm.Recorder("trx_map", {
        **tm.base_meta(dev, args),
        "band": args.band,
        "bias1_names": ["v1", "h1"],
        "bias2_names": ["v2", "h2"],
        "bias1_da": [ch["v1"], ch["h1"]],
        "bias2_da": [ch["v2"], ch["h2"]],
        "sa_params": dev.sa_params(),
    }, columns=["bias1_mV", "bias2_mV"])

    dev.select_band(args.band)
    dev.close_band(args.band)

    if args.auto:
        tm.ask_yes("Are you ready")
    try:
        for v1 in vol1:
            dev.set_bias(v1=float(v1), h1=float(v1))
            if not args.auto:
                tm.ask_yes("Are you ready")
            for v2 in vol2:
                print(f"bias1 = {v1:.3f}, bias2 = {v2:.3f} mV")
                dev.set_bias(v2=float(v2), h2=float(v2))
                time.sleep(args.settle)
                rec.add(dev.fresh_trace(), bias1_mV=round(float(v1), 4), bias2_mV=round(float(v2), 4))
    finally:
        dev.zero_bias()
        print("All SIS biases set to 0. Set the operating values again.")

    tm.finish(rec)


if __name__ == "__main__":
    main()
