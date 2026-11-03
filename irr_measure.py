#!/usr/bin/env python3
"""
Image rejection ratio of both polarizations at one set of SIS biases.

HOT and COLD loads are placed by hand; the test signal comes from the Phase
Matrix FSW-0020 (IrrSignalGenerator of TZ-controller).

Usage (in the folder that holds this script):
    uv run python irr_measure.py V1 H1 H2 V2 \
        [--usb-freq 16.609] [--lsb-freq 14.609] [--soft-average N]
    uv run python irr_measure.py --point N ...   biases from row N of irr_points.csv

    --point N     take V1 H1 H2 V2 from the row "point = N" of irr_points.csv
                  next to this script. Columns: point, v1, h1, h2, v2 [mV].
                  Copy irr_points_example.csv to irr_points.csv and write one
                  row per point of a map, then call the script once per point.

    V1 H1 H2 V2   SIS biases [mV]
    --usb-freq, --lsb-freq
                  test signal SG frequency [GHz] (asked if omitted).
                  The SG output is multiplied by 6 before the receiver.
    --soft-average N
                  average N sweeps on the PC instead of in the analyzer
                  (off by default). The analyzer's averaging is turned off,
                  every sweep is recorded, and the analysis gives the spread
                  of the IRR between sweeps. See Devices.new_sweeps.

Sequence: set the biases and turn the test signal off; with the COLD and then
the HOT load, for HU, HL, VU, VL select the IF band, wait 10 s, take one
trace, close the path. Then with the USB and then the LSB test signal on, the
same four bands. The biases are left as set.

"y" is asked only where the operator acts (the OLD MMC switch to COLD, then
to HOT); --ask-each-band asks before every band and test signal too. The
analyzer averaging is restarted and completed for every trace
(Devices.fresh_trace), which the TZ manual asks the operator to confirm by hand.
"""

import argparse
import csv
import pathlib
import time

import rx_performance_common as tm


# Bias table for --point: columns point, v1, h1, h2, v2 [mV]
POINTS_FILE = pathlib.Path(__file__).resolve().parent / "irr_points.csv"


def read_point(n: int) -> dict:
    with open(POINTS_FILE) as f:
        for row in csv.DictReader(f):
            if int(row["point"]) == n:
                return {name: float(row[name]) for name in tm.BIAS_NAMES}
    raise SystemExit(f"No point {n} in {POINTS_FILE.name}.")


def ask_freq(label: str) -> float:
    while True:
        s = input(f"{label} test signal SG freq [GHz]: ")
        try:
            return float(s)
        except ValueError:
            continue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("bias", nargs="*", type=float, metavar="MV",
                   help="SIS biases V1 H1 H2 V2 [mV] (or --point)")
    p.add_argument("--point", type=int,
                   help=f"take V1 H1 H2 V2 from row N of {POINTS_FILE.name} instead of typing them")
    p.add_argument("--usb-freq", type=float)
    p.add_argument("--lsb-freq", type=float)
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF band [s]")
    p.add_argument("--ask-each-band", action="store_true",
                   help="wait for y before every band and test signal")
    p.add_argument("--soft-average", type=int, default=0, metavar="N",
                   help="average N separate sweeps on the PC (default: analyzer averaging)")
    tm.add_common_args(p)
    args = p.parse_args()

    if args.point is not None:
        if args.bias:
            raise SystemExit("Give either the four biases or --point, not both.")
        bias = read_point(args.point)
        print(f"point {args.point} of {POINTS_FILE.name}: "
              + "  ".join(f"{n} {bias[n]:.2f}" for n in tm.BIAS_NAMES) + " mV")
    elif len(args.bias) == 4:
        bias = dict(zip(tm.BIAS_NAMES, args.bias))
    else:
        raise SystemExit("Give the four biases V1 H1 H2 V2, or --point N.")

    dev = tm.start(args, use={"sis", "sa", "switch", "irr"})
    if args.soft_average < 0:
        raise SystemExit("--soft-average must be 0 or more.")
    if not args.keep_sa:
        dev.setup_sa("irr", average_off=args.soft_average > 0)

    rec = tm.Recorder("irr_measure", {
        **tm.base_meta(dev, args),
        "bias_mV": bias,
        "point": args.point,
        "sa_params": dev.sa_params(),
        "soft_average": args.soft_average,
    }, columns=["load", "band", "signal_GHz", "sweep", "t_s"])

    def sweep_bands(load: str, signal) -> None:
        for band in tm.BANDS:
            dev.select_band(band)
            print("setting", band)
            time.sleep(args.settle)
            if args.ask_each_band:
                tm.ask_yes("ready?")
            if args.soft_average:
                for k, (t, trace) in enumerate(dev.new_sweeps(args.soft_average)):
                    rec.add(trace, load=load, band=band, signal_GHz=signal, sweep=k, t_s=t)
            else:
                rec.add(dev.fresh_trace(), load=load, band=band, signal_GHz=signal, sweep=0, t_s="")
            dev.close_band(band)

    try:
        dev.set_bias(**bias)
        dev.set_irr(output=False)

        for load in ["cold", "hot"]:
            print("--------", load, "--------")
            tm.ask_yes(f"Set the OLD MMC switch to {load.upper()}, then start")
            tm.set_load(load)
            sweep_bands(load, "")

        freqs = {"usbtest": args.usb_freq, "lsbtest": args.lsb_freq}
        for test in ["usbtest", "lsbtest"]:
            print("--------", test, "--------")
            f = freqs[test] if freqs[test] is not None else ask_freq(test[:3].upper())
            dev.set_irr(freq_ghz=f)
            dev.set_irr(output=True)
            if args.ask_each_band:
                tm.ask_yes("measurement start")
            else:
                print(f"test signal on: {f} GHz")
            tm.set_load("hot")
            sweep_bands(test, f)
            dev.set_irr(output=False)
    finally:
        dev.irr.teardown()

    tm.finish(rec)


if __name__ == "__main__":
    main()
