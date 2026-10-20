#!/usr/bin/env python3
"""
Receiver noise temperature versus LO attenuator current, HL and VL bands.

Port of the ROS script yfactor_with_search_loatt_level_direction-ver2.py.
The chopper must be rotating.

Usage:
    cd ~/tz_workspace && uv run python loatt_search.py START STOP NUM FREQ \
        [--bias V1 H1 H2 V2]

    START STOP NUM  attenuator currents [mA], numpy.linspace(START, STOP, NUM).
                    The ROS script called the third argument "step" but used
                    it as the number of points; here it is named for what it is.
    FREQ            1st LO SG frequency, recorded as a label
    --bias          SIS biases to hold during the scan (default: tuned values
                    of device_config_ros.toml; ROS asked for them)

Sequence (as in ROS): set the SIS biases, wait for "y"; for each current set
H, wait 60 s, set V; for HL and VL open and close the IF path, wait 10 s and
take traces. At the end the operator types the H and V currents to keep.
"""

import argparse

import numpy as np

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("start", type=float)
    p.add_argument("stop", type=float)
    p.add_argument("num", type=int)
    p.add_argument("freq")
    p.add_argument("--bias", nargs=4, type=float, metavar=("V1", "H1", "H2", "V2"))
    p.add_argument("--att-settle", type=float, default=60.0,
                   help="wait after setting the H attenuator [s] (ROS: 60)")
    p.add_argument("--settle", type=float, default=10.0,
                   help="wait after switching the IF path [s] (ROS: 10)")
    p.add_argument("--traces", type=int, default=1,
                   help="traces per point (ROS logged for 20 s)")
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sis", "sa", "switch", "loatt"})
    tuned = rig.cfg["sis_bias_setter"]["tuned"]
    bias = dict(zip(tm.BIAS_NAMES, args.bias)) if args.bias else {n: tuned[n] for n in tm.BIAS_NAMES}
    att_list = np.linspace(args.start, args.stop, args.num)
    if not args.keep_sa:
        rig.setup_sa("sa_chopper")

    rec = tm.Recorder("loatt_search", {
        **tm.base_meta(rig, args),
        "freq": args.freq,
        "bias_mV": bias,
        "sa_params": rig.sa_params(),
    }, columns=["att_mA", "band", "repeat"])

    rig.set_bias(**bias)
    print("SIS bias:", bias)
    tm.ask_yes("ready?")

    for att in att_list:
        att = round(float(att), 4)
        print(f"now setting att. = {att}  (please wait {args.att_settle:.0f} s)")
        rig.set_loatt("H", att)
        tm.sleep(rig, args.att_settle)
        rig.set_loatt("V", att)
        for band in ["HL", "VL"]:
            rig.open_band(band)
            rig.close_band(band)
            print("current measurement band:", band)
            tm.sleep(rig, args.settle)
            for k in range(args.traces):
                rec.add(rig.trace(), att_mA=att, band=band, repeat=k)

    tm.finish(rec)

    print("finished att. param search... please set att.")
    att_h = 23.0 if args.yes else float(input("set H-pol. att [mA]: "))
    att_v = 23.0 if args.yes else float(input("set V-pol. att [mA]: "))
    print(f"setting H att. at {att_h} mA (please wait {args.att_settle:.0f} s)")
    rig.set_loatt("H", att_h)
    tm.sleep(rig, args.att_settle)
    print(f"setting V att. at {att_v} mA (please wait {args.att_settle:.0f} s)")
    rig.set_loatt("V", att_v)
    tm.sleep(rig, args.att_settle)
    print("setting finished")


if __name__ == "__main__":
    main()
