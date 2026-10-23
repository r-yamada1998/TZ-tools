#!/usr/bin/env python3
"""
Ask the IF switch driver (Agilent 11713B) whether it receives the commands and
drives the switch.

Written after pol_test on 2026-10-22: the four IF bands read within 0.05 dB
of each other and did not respond to any bias change, although the bias
monitors confirmed the biases. The band selection commands are sent without
any answer, so a missing error proved nothing. This script asks the 11713B
for its identity, the drive supply of bank 1, the state of every path, and
the relay cycle counters of the four IF paths (105-108); then it selects each
band in turn as the measurement scripts do (open, 0.1 s, close) and asks
again. A counter that goes up shows that the 11713B really drove that path.
The analyzer level is read after each selection (one triggered sweep, three
times).

Usage (in the folder that holds this script):
    uv run python if_switch_check.py

Run it with the LO on as well: then the four bands should read differently
if the switch really selects them.

Changes nothing but the selected IF band and, for the duration of the run,
the analyzer settings (chopper settings, back to continuous sweep at the end).
Output: on screen, and if_switch_check_<YYMMDD_HHMMSS>.txt.

On 2026-10-23 the first query after *IDN? (CONFigure:BANK1?) timed out, and
every later query failed with "cannot read from timed out object": once a
read times out, the socket file of ogameasure.ethernet cannot be read again.
A failed query now reconnects and asks SYST:ERR?, so that each query gets its
own answer and the reason for a missing answer is seen.
"""

import argparse
import datetime

import numpy as np

import rx_performance_common as tm

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


def ask(rig, cmd: str) -> str:
    """One query to the 11713B through the GPIB adapter."""
    if rig.dry_run:
        return "(dry run)"
    com = rig.switch.sw.com
    try:
        com.send(cmd)
        return com.readline().strip()
    except Exception as e:  # a timeout here is itself the answer
        reconnect(com)
        try:
            com.send("SYST:ERR?")
            err = com.readline().strip()
        except Exception:
            reconnect(com)
            err = "no answer either"
        return f"no answer ({type(e).__name__}: {e}); SYST:ERR? -> {err}"


def reconnect(com) -> None:
    """A timed-out read leaves the socket file unreadable; start a new one."""
    try:
        com.close()
    except Exception:
        pass
    com.open()


def level(rig) -> float:
    return float(np.mean([np.mean(rig.single_trace()) for _ in range(3)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    tm.add_common_args(p)
    args = p.parse_args()

    rig = tm.start(args, use={"sa", "switch"})
    out = Out()
    out.say(f"if_switch_check {datetime.datetime.now().isoformat()}  dry run: {args.dry_run}")
    out.say(f"  *IDN?                  {ask(rig, '*IDN?')}")
    out.say(f"  errors before          {ask(rig, 'SYST:ERR?')}")
    out.say(f"  bank 1 drive supply    {ask(rig, 'CONFigure:BANK1?')}")
    out.say(f"  bank 1 TTL drive       {ask(rig, 'CONFigure:BANK1:TTL?')}")
    out.say(f"  open?  101-110         {ask(rig, ':ROUTe:OPEn? ' + PATHS)}")
    out.say(f"  close? 101-110         {ask(rig, ':ROUTe:CLOSe? ' + PATHS)}")
    out.say(f"  relay cycles 105-108   {ask(rig, ':DIAGnostic:RELay:CYCles? ' + IF_PATHS)}")
    out.say(f"  errors                 {ask(rig, 'SYST:ERR?')}")

    rig.setup_sa("sa_chopper", average_off=True)
    rig.set_continuous(False)
    try:
        for band in tm.BANDS:
            rig.open_band(band)
            path = rig.m["if_switch_path"][band]
            lv = level(rig)
            out.say(f"== {band} selected (path {path}) ==")
            out.say(f"  open?  101-110         {ask(rig, ':ROUTe:OPEn? ' + PATHS)}")
            out.say(f"  relay cycles 105-108   {ask(rig, ':DIAGnostic:RELay:CYCles? ' + IF_PATHS)}")
            out.say(f"  analyzer level         {lv:.3f} dBm")
        out.say(f"  errors                 {ask(rig, 'SYST:ERR?')}")
    finally:
        rig.set_continuous(True)
        out.say("  analyzer back to continuous sweep")
    out.say(f"Saved: {out.path}")


if __name__ == "__main__":
    main()
