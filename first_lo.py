#!/usr/bin/env python3
"""
Switch the 1st LO (AnaPico APSYN420) on or off. tz_status.py reads its state.

"on" sends CW, the external 10 MHz reference, reference output on, the
frequency and output on. The power is not sent, so the power set on the
generator stays.

Usage (in the folder that holds this script):
    uv run python first_lo.py on              signal_generator.defaults.freq of the
                                              device config (15.609 GHz, x6 = 93.654 GHz)
    uv run python first_lo.py on 15.8         another frequency [GHz]
    uv run python first_lo.py off

After "on", the frequency read back must equal the one set (to the 1 mHz
resolution of the generator) and the reference must be locked; after "off",
the output must read back off. Otherwise NG is printed and the exit status is 1.

Before switching the 1st LO, put the LO Att. to 70 70 (receiver developer);
initialize.py does this in the right order.
"""

import argparse
import time

import rx_performance_common as tm

# The reference lock is re-acquired after the reference is set; the
# programmer's manual asks for at least 3 s before reading the lock state.
LOCK_WAIT = 3.0      # s, before the first reading
LOCK_TIMEOUT = 15.0  # s, then give up and show what was read

# POW? returns the power setting (0 dBm after *RST), not the fixed output
# (+23 dBm nominal), so it is not shown.
HIDDEN = {"power_dbm"}


def _fields(status: dict) -> list:
    """Status as "name = value" strings; the frequency in GHz to 1 Hz."""
    out = []
    for k, v in status.items():
        if k in HIDDEN:
            continue
        if k == "freq_hz" and v is not None:
            out.append(f"freq_ghz = {float(v) / 1e9:.9f}")
        else:
            out.append(f"{k} = {v}")
    return out


def report(dev):
    """Print the generator's state and return it (None if it could not be read).
    A failed query must not stop the caller."""
    try:
        status = dev.lo1.status()
    except Exception as e:
        print(f"1st LO: state could not be read ({e!r})")
        return None
    print("1st LO: " + ", ".join(_fields(status)))
    return status


def check(status, freq_ghz: float) -> bool:
    """OK only if the frequency read back is the one set and the reference is locked."""
    if status is None:
        print("1st LO: NG (state could not be read)")
        return False
    read_hz = float(status["freq_hz"])
    set_hz = round(freq_ghz * 1e9, 3)
    problems = []
    if abs(read_hz - set_hz) >= 5e-4:
        problems.append(f"frequency {read_hz / 1e9:.9f} GHz is not the set {freq_ghz:.9f} GHz")
    if not status["ref_locked"]:
        problems.append("reference not locked")
    if not status["output"]:
        problems.append("output off")
    if problems:
        print("1st LO: NG (" + "; ".join(problems) + ")")
        return False
    print("1st LO: OK")
    return True


def check_off(status) -> bool:
    """OK only if the output reads back off."""
    if status is None:
        print("1st LO: NG (state could not be read)")
        return False
    if status.get("output"):
        print("1st LO: NG (output still on)")
        return False
    print("1st LO: OK (output off)")
    return True


def report_after_on(dev, freq_ghz: float) -> bool:
    """Wait for the reference lock (up to LOCK_TIMEOUT), report and check."""
    time.sleep(LOCK_WAIT)
    t0 = time.monotonic()
    while time.monotonic() - t0 < LOCK_TIMEOUT:
        try:
            if dev.lo1.status().get("ref_locked"):
                break
        except Exception:
            break
        time.sleep(1.0)
    return check(report(dev), freq_ghz)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("state", choices=["on", "off"])
    p.add_argument("freq", type=float, nargs="?", default=None,
                   help="frequency [GHz] with on (default: signal_generator.defaults.freq)")
    tm.add_common_args(p)
    args = p.parse_args()
    if args.freq is None:
        args.freq = tm.tuned_lo1_ghz(tm.load_configs(args.device_config)[1])

    # APSYN420 range; not a limit of the x6 multiplier
    if args.state == "on" and not 0.0 < args.freq <= 20.0:
        raise SystemExit(f"Aborted: {args.freq} GHz is outside 0-20 GHz.")
    dev = tm.start(args, use={"lo1"})
    ok = True
    try:
        if args.state == "on":
            dev.lo1_on(args.freq)
            ok = report_after_on(dev, args.freq)
        else:
            dev.lo1_off()
            ok = check_off(report(dev))
    finally:
        dev.disconnect_lo1()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
