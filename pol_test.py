#!/usr/bin/env python3
"""
Which IF bands each SIS bias terminal feeds: settles the polarization naming.

v1 and h2 are swapped between the ROS writer (v1 = DA ch7) and the ROS reader
and the new system (v1 = DA ch5); which one matches the physical polarization
is not known. This test answers it directly. With the LO on, the biases are
set to the operating values; then each DA terminal in turn is set to 0 mV
while the output level of all four IF bands is read. A terminal that feeds the
H mixers changes HU and HL, one that feeds V changes VU and VL.

Usage (in the folder that holds this script):
    uv run python pol_test.py [--settle 4] [--read 2]

Needs: LO on, SIS bias switch on OPERATE. The chopper may stand still. Biases
are left at the operating values of device_config_ros.toml.

Output: pol_test_<YYMMDD_HHMMSS>/levels.csv and the table on screen.
"""

import argparse
import datetime
import pathlib
import time

import numpy as np

import rx_performance_common as tm


def level(rig, band: str, settle: float, read: float) -> float:
    """Mean output of one IF band [dBm], over traces read for `read` seconds."""
    rig.open_band(band)
    tm.sleep(rig, settle)
    vals = []
    t0 = time.monotonic()
    while True:
        vals.append(float(np.mean(rig.trace())))
        if rig.dry_run or time.monotonic() - t0 > read:
            break
    return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--settle", type=float, default=4.0, help="wait after switching the IF band [s]")
    p.add_argument("--read", type=float, default=2.0, help="read traces for this long per band [s]")
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sis", "sa", "switch"})
    tuned = dict(rig.cfg["sis_bias_setter"]["tuned"])
    ch = rig.cfg["sis_bias_setter"]["channel"]
    rig.setup_sa("sa_chopper", average_off=True)
    tm.ask_yes("LO on and SIS bias switch on OPERATE. Start")

    out_dir = pathlib.Path(".") / f"pol_test_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    out_dir.mkdir()
    rows = []
    try:
        rig.set_bias(**tuned)
        tm.sleep(rig, 2.0)
        base = {b: level(rig, b, args.settle, args.read) for b in tm.BANDS}
        rows += [["none", "", b, round(base[b], 3), 0.0] for b in tm.BANDS]
        print("operating values: " + ", ".join(f"{b} {v:.2f} dBm" for b, v in base.items()))
        if max(base.values()) - min(base.values()) < 0.1:
            # On 2026-10-22 (LO off) all four bands agreed within 0.05 dB.
            print("WARNING: all four bands agree within 0.1 dB. The IF switch may not be switching;"
                  " then every row below changes all bands alike.")
        table = {}
        for name in tm.BIAS_NAMES:
            rig.set_bias(**{name: 0.0})
            tm.sleep(rig, 1.0)
            lv = {b: level(rig, b, args.settle, args.read) for b in tm.BANDS}
            rig.set_bias(**{name: tuned[name]})
            table[name] = {b: lv[b] - base[b] for b in tm.BANDS}
            rows += [[name, ch[name], b, round(lv[b], 3), round(lv[b] - base[b], 3)] for b in tm.BANDS]
            print(f"{name} ({ch[name]}) at 0 mV: " + ", ".join(f"{b} {d:+.2f} dB" for b, d in table[name].items()))
    finally:
        rig.set_bias(**tuned)
        print("SIS biases back at the operating values: " + ", ".join(f"{k} = {v}" for k, v in tuned.items()))

    with (out_dir / "levels.csv").open("w") as f:
        f.write("zeroed,da_channel,band,level_dBm,change_dB\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print("\nChange of the output when one terminal is set to 0 mV [dB]")
    print("name  DA     " + "  ".join(f"{b:>6}" for b in tm.BANDS) + "   feeds")
    for name, d in table.items():
        h = abs(d["HU"]) + abs(d["HL"])
        v = abs(d["VU"]) + abs(d["VL"])
        feeds = "H" if h > 2 * v else "V" if v > 2 * h else "unclear"
        print(f"{name:<5} {ch[name]:<6} " + "  ".join(f"{d[b]:+6.2f}" for b in tm.BANDS) + f"   {feeds}")
    print(f"\nSaved: {out_dir}/levels.csv")


if __name__ == "__main__":
    main()
