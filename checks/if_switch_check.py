#!/usr/bin/env python3
"""
Ask the IF Switch Controller (Agilent 11713B) whether it receives the
commands and drives the switch.

The band selection commands get no answer, so a missing error proves nothing.
This script asks the controller for its identity, the drive supply of bank 1,
the state of every path, and the relay cycle counters of the four IF paths
(105-108); then it selects each band in turn as the measurement scripts do
(open, 0.1 s, close) and asks again. A counter that goes up shows that the
controller really drove that path. The analyzer level is read after each
selection (one triggered sweep, three times).

Usage (in the TZ-tools folder):
    uv run python checks/if_switch_check.py

Run it with the LO on as well: then the four bands should read differently
if the switch really selects them.

Changes nothing but the selected IF band and, for the duration of the run,
the analyzer settings (chopper settings, back to continuous sweep at the end).
Output: on screen, and if_switch_check_<YYMMDD_HHMMSS>.txt.

A query that times out is answered by IFSwitch.query with the controller's
SYST:ERR?, so the reason for a missing answer is seen.
"""

import argparse
import datetime
import pathlib

import numpy as np

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rx_performance_common as tm  # noqa: E402

# the explicit list form, as the ogameasure driver writes it
PATHS = "(@101,102,103,104,105,106,107,108,109,110)"
IF_PATHS = "(@105,106,107,108)"


class Out:
    def __init__(self):
        self.path = f"if_switch_check_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.txt"
        self.lines = []

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.lines.append(text)
        with open(self.path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def ask(dev, cmd: str) -> str:
    """One query to the controller through the GPIB adapter."""
    try:
        return dev.switch.query(cmd)
    except TimeoutError as e:  # a timeout here is itself the answer
        return f"no answer ({e})"


def level(dev) -> float:
    return float(np.mean([np.mean(dev.single_trace()) for _ in range(3)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use={"sa", "switch"})
    out = Out()
    out.say(f"if_switch_check {datetime.datetime.now().isoformat()}  dry run: {args.dry_run}")
    out.say(f"  *IDN?                  {ask(dev, '*IDN?')}")
    out.say(f"  errors before          {ask(dev, 'SYST:ERR?')}")
    out.say(f"  bank 1 drive supply    {ask(dev, 'CONFigure:BANK1?')}")
    out.say(f"  bank 1 TTL drive       {ask(dev, 'CONFigure:BANK1:TTL?')}")
    out.say(f"  open?  101-110         {ask(dev, ':ROUTe:OPEn? ' + PATHS)}")
    out.say(f"  close? 101-110         {ask(dev, ':ROUTe:CLOSe? ' + PATHS)}")
    out.say(f"  relay cycles 105-108   {ask(dev, ':DIAGnostic:RELay:CYCles? ' + IF_PATHS)}")
    out.say(f"  errors                 {ask(dev, 'SYST:ERR?')}")

    dev.setup_sa("chopper", average_off=True)
    dev.set_continuous(False)
    try:
        for band in tm.BANDS:
            dev.select_band(band)
            path = dev.m["if_switch"]["path"][band]
            lv = level(dev)
            out.say(f"== {band} selected (path {path}) ==")
            out.say(f"  open?  101-110         {ask(dev, ':ROUTe:OPEn? ' + PATHS)}")
            out.say(f"  relay cycles 105-108   {ask(dev, ':DIAGnostic:RELay:CYCles? ' + IF_PATHS)}")
            out.say(f"  analyzer level         {lv:.3f} dBm")
        out.say(f"  errors                 {ask(dev, 'SYST:ERR?')}")
    finally:
        dev.set_continuous(True)
        out.say("  analyzer back to continuous sweep")
    out.say(f"Saved: {out.path}")


if __name__ == "__main__":
    main()
