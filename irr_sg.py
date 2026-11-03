#!/usr/bin/env python3
"""
Switch the IRR test signal (Phase Matrix FSW-0020) on at a frequency, or off.

The generator has no front panel, so this is the way to drive it by hand
(TZ manual 3.5).

Usage (in the folder that holds this script):
    uv run python irr_sg.py on 14.609     LSB test signal (87.654 GHz after x6)
    uv run python irr_sg.py on 16.609     USB test signal (99.654 GHz after x6)
    uv run python irr_sg.py off

The generator does not answer queries, so the state cannot be read back;
check the test signal on the spectrum analyzer.
"""

import argparse

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("state", choices=["on", "off"])
    p.add_argument("freq", type=float, nargs="?", help="SG frequency [GHz], needed with on")
    tm.add_common_args(p)
    args = p.parse_args()

    if args.state == "on":
        if args.freq is None:
            raise SystemExit("Give the frequency, e.g. irr_sg.py on 14.609")
        # FSW-0020 range
        if not 2.0 <= args.freq <= 20.0:
            raise SystemExit(f"Aborted: {args.freq} GHz is outside 2-20 GHz.")
    dev = tm.start(args, use={"irr"})
    try:
        if args.state == "on":
            dev.set_irr(freq_ghz=args.freq)
            dev.set_irr(output=True)
            print(f"Test signal on: {args.freq} GHz (x6 = {6 * args.freq:.3f} GHz)")
        else:
            dev.set_irr(output=False)
            print("Test signal off")
    finally:
        dev.irr.close()


if __name__ == "__main__":
    main()
