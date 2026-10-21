#!/usr/bin/env python3
"""
Image rejection ratio of both polarizations at one set of SIS biases.

Port of the ROS script irr_measurement2.py. HOT and COLD loads are placed by
hand; the test signal comes from the Phase Matrix FSW-0020.

Usage (in the folder that holds this script):
    uv run python irr_measure.py V1 H1 H2 V2 \
        [--usb-freq 16.609] [--lsb-freq 14.609] [--soft-average N]

    V1 H1 H2 V2   SIS biases [mV], in the order of set_sis_vol_direction.py
    --usb-freq, --lsb-freq
                  test signal SG frequency [GHz] (asked if omitted, as in ROS).
                  The SG output is multiplied by 6 before the receiver.
    --soft-average N
                  average N sweeps on the PC instead of in the analyzer
                  (not in ROS; off by default). The analyzer's averaging is
                  turned off, every sweep is recorded, and the analysis gives
                  the spread of the IRR between sweeps. See Rig.new_sweeps.

Sequence (as in ROS): set the biases and turn the test signal off; with the
COLD and then the HOT load, for HU, HL, VU, VL open the IF path, wait 10 s,
wait for "y", take one trace, close the path. Then with the USB and then the
LSB test signal on, the same four bands. The biases are left as set.
"""

import argparse

import rx_performance_common as tm


def ask_freq(label: str) -> float:
    while True:
        s = input(f"{label} test signal SG freq [GHz]: ")
        try:
            return float(s)
        except ValueError:
            continue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    for name in tm.BIAS_NAMES:
        p.add_argument(name, type=float, help=f"SIS bias {name} [mV]")
    p.add_argument("--usb-freq", type=float)
    p.add_argument("--lsb-freq", type=float)
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF path [s] (ROS: 10)")
    p.add_argument("--soft-average", type=int, default=0, metavar="N",
                   help="average N separate sweeps on the PC (default: analyzer averaging)")
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sis", "sa", "switch", "irr"})
    bias = {name: getattr(args, name) for name in tm.BIAS_NAMES}
    if args.soft_average < 0:
        raise SystemExit("--soft-average must be 0 or more.")
    if not args.keep_sa:
        rig.setup_sa("sa_irr", average_off=args.soft_average > 0)

    rec = tm.Recorder("irr_measure", {
        **tm.base_meta(rig, args),
        "bias_mV": bias,
        "sa_params": rig.sa_params(),
        "soft_average": args.soft_average,
    }, columns=["load", "band", "signal_GHz", "sweep", "t_s"])

    def sweep_bands(load: str, signal) -> None:
        for band in tm.BANDS:
            rig.open_band(band)
            print("setting", band)
            tm.sleep(rig, args.settle)
            tm.ask_yes("ready?")
            if args.soft_average:
                for k, (t, trace) in enumerate(rig.new_sweeps(args.soft_average)):
                    rec.add(trace, load=load, band=band, signal_GHz=signal, sweep=k, t_s=t)
            else:
                rec.add(rig.fresh_trace(), load=load, band=band, signal_GHz=signal, sweep=0, t_s="")
            rig.close_band(band)

    try:
        rig.set_bias(**bias)
        rig.set_irr_output(False)

        # The ROS file names called these the black body temperatures.
        for load in ["cold", "hot"]:
            print("--------", load, "--------")
            tm.ask_yes(f"Set the OLD MMC switch to {load.upper()}, then start")
            rig.load = load
            sweep_bands(load, "")

        freqs = {"usbtest": args.usb_freq, "lsbtest": args.lsb_freq}
        for test in ["usbtest", "lsbtest"]:
            print("--------", test, "--------")
            f = freqs[test] if freqs[test] is not None else ask_freq(test[:3].upper())
            rig.set_irr_freq(f)
            rig.set_irr_output(True)
            tm.ask_yes("measurement start")
            rig.load = "hot"
            sweep_bands(test, f)
            rig.set_irr_output(False)
    finally:
        rig.set_irr_output(False)
        rig.close_irr()

    tm.finish(rec)


if __name__ == "__main__":
    main()
