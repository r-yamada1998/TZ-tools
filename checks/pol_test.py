#!/usr/bin/env python3
"""
Which IF bands each SIS bias terminal feeds: checks the polarization naming.

With the LO on, the biases are set to the operating values; then each DA
terminal in turn is set to 0 mV while the output level of all four IF bands
is read. A terminal that feeds the H mixers changes HU and HL, one that feeds
V changes VU and VL. On 2026-10-23 it found DA ch6, ch7 on H and ch5, ch8 on V.

Usage (in the TZ-tools folder):
    uv run python checks/pol_test.py [--settle 4] [--read 2]
    uv run python checks/pol_test.py --bias 8.5 8.5 8.5 8.5      LO off

Needs: SIS bias switch on OPERATE. The chopper may stand still. Without
--bias: LO on, and the biases are left at the operating values. With --bias
the LO may be off: above the gap voltage the junction is no longer
superconducting, so its IF output differs from that at 0 mV even without LO.
This also shows whether the IF switch switches. The biases are left at 0 mV.

The bias monitors (voltage and current of each junction, AD ch9-16) are read
after every change, so that the table also shows that the biases were really
applied. They are labelled by DA terminal.

Output: pol_test_<YYMMDD_HHMMSS>/levels.csv, monitors.csv and the table on screen.
"""

import argparse
import datetime
import pathlib
import time

import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rx_performance_common as tm  # noqa: E402

NAMES = ["v1", "h1", "h2", "v2"]


def open_monitor(dev) -> list:
    """All four bias monitors, voltage channels registered first."""
    return dev.setup_monitors([f"{n}_v" for n in NAMES] + NAMES)


def read_monitor(dev, names: list, da_of: dict) -> dict:
    """{DA terminal: (mV, uA)}"""
    r = dev.read_monitors(names)
    return {da_of[n]: (r[f"{n}_v"], r[n]) for n in NAMES}


def level(dev, band: str, settle: float, read: float) -> float:
    """Mean output of one IF band [dBm], over traces read for `read` seconds."""
    dev.select_band(band)
    time.sleep(settle)
    vals = []
    t0 = time.monotonic()
    while True:
        vals.append(float(np.mean(dev.trace())))
        if time.monotonic() - t0 > read:
            break
    return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--settle", type=float, default=4.0, help="wait after switching the IF band [s]")
    p.add_argument("--read", type=float, default=2.0, help="read traces for this long per band [s]")
    p.add_argument("--bias", nargs=4, type=float, metavar="MV",
                   help="V1 H1 H2 V2 [mV] instead of the operating values; biases end at 0")
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use={"sis", "sa", "switch"})
    tuned = tm.tuned_bias(dev.cfg)
    if args.bias:
        tuned = dict(zip(tm.BIAS_NAMES, args.bias))
    final = {name: 0.0 for name in tm.BIAS_NAMES} if args.bias else tuned
    ch = dev.cfg["sis_bias_setter"]["channel"]
    dev.setup_sa("chopper", average_off=True)
    dev.open({"reader"})
    names = open_monitor(dev)
    da_of = ch
    mon_rows = []

    def monitor(label: str) -> None:
        m = read_monitor(dev, names, da_of)
        if m:
            print(f"  monitors ({label}): " + ", ".join(f"{d} {v:.2f} mV {i:.1f} uA" for d, (v, i) in sorted(m.items())))
            mon_rows.extend([label, d, round(v, 3), round(i, 2)] for d, (v, i) in sorted(m.items()))
    print("biases for the test: " + ", ".join(f"{k} = {v} mV" for k, v in tuned.items()))
    tm.ask_yes(("LO may be off" if args.bias else "LO on") + " and SIS bias switch on OPERATE. Start")

    out_dir = pathlib.Path(".") / f"pol_test_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    out_dir.mkdir()
    rows = []
    try:
        dev.set_bias(**tuned)
        time.sleep(2.0)
        monitor("all set")
        base = {b: level(dev, b, args.settle, args.read) for b in tm.BANDS}
        rows += [["none", "", b, round(base[b], 3), 0.0] for b in tm.BANDS]
        print("operating values: " + ", ".join(f"{b} {v:.2f} dBm" for b, v in base.items()))
        if max(base.values()) - min(base.values()) < 0.1:
            # On 2026-10-22 (LO off) all four bands agreed within 0.05 dB.
            print("WARNING: all four bands agree within 0.1 dB. The IF switch may not be switching;"
                  " then every row below changes all bands alike.")
        table = {}
        for name in tm.BIAS_NAMES:
            dev.set_bias(**{name: 0.0})
            time.sleep(1.0)
            monitor(f"{name} at 0")
            lv = {b: level(dev, b, args.settle, args.read) for b in tm.BANDS}
            dev.set_bias(**{name: tuned[name]})
            table[name] = {b: lv[b] - base[b] for b in tm.BANDS}
            rows += [[name, ch[name], b, round(lv[b], 3), round(lv[b] - base[b], 3)] for b in tm.BANDS]
            print(f"{name} ({ch[name]}) at 0 mV: " + ", ".join(f"{b} {d:+.2f} dB" for b, d in table[name].items()))
    finally:
        dev.set_bias(**final)
        print("SIS biases left at: " + ", ".join(f"{k} = {v}" for k, v in final.items()))
        time.sleep(0.5)
        monitor("end")
        dev.reader.teardown()

    with (out_dir / "monitors.csv").open("w") as f:
        f.write("state,da_channel,v_mV,i_uA\n")
        for r in mon_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with (out_dir / "levels.csv").open("w") as f:
        f.write("zeroed,da_channel,band,level_dBm,change_dB\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print("\nChange of the output when one terminal is set to 0 mV [dB]")
    print("name  DA     " + "  ".join(f"{b:>6}" for b in tm.BANDS) + "   feeds")
    for name, d in table.items():
        h = abs(d["HU"]) + abs(d["HL"])
        v = abs(d["VU"]) + abs(d["VL"])
        # h and v add two bands, so 1.0 means about 0.5 dB per band; the scatter
        # of the levels was up to 0.07 dB on 2026-10-22.
        feeds = "none" if max(h, v) < 1.0 else "H" if h > 2 * v else "V" if v > 2 * h else "unclear"
        print(f"{name:<5} {ch[name]:<6} " + "  ".join(f"{d[b]:+6.2f}" for b in tm.BANDS) + f"   {feeds}")
    print(f"\nSaved: {out_dir}/levels.csv")


if __name__ == "__main__":
    main()
