#!/usr/bin/env python3
"""
Read the state of TZ back without changing anything.

Meant for a second terminal while initialize.py waits for "finished",
or at any time. The SIS monitors are read first, so they are shown even when
the 1st LO cannot be reached. Nothing is sent to the DA board or the LO Att.

    SIS mixers  junction voltage and current from the AD monitors,
                next to the tuned values
    1st LO      frequency, output, reference source and lock
    LO Att.     the last currents set from TZ-tools (set_loatt.py,
                initialize.py, the measurement scripts). The LO Att.
                cannot be asked for its current, so this is not a reading;
                a value set by another program (initialize.py) is not shown.

Usage (in the folder that holds this script):
    uv run python tz_status.py
"""

import argparse

import first_lo
import rx_performance_common as tm


def read_sis(dev) -> dict:
    """{mixer: (voltage [mV], current [uA])}, one mixer at a time."""
    out = {}
    for name in tm.BIAS_NAMES:
        # Voltage channel first: a large signal registered last leaks into
        # the first channel of the next sampling.
        names = dev.setup_monitors([f"{name}_v", name])
        try:
            r = dev.read_monitors(names)
        finally:
            # Stop the sampling before the next pair; otherwise the new
            # channels are not taken and every row shows the first pair.
            dev.reader.teardown()
        out[name] = (r[f"{name}_v"], r[name])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use=set())
    tuned = dev.cfg["sis_bias_setter"]["tuned"]
    ch = dev.cfg["sis_bias_setter"]["channel"]
    path = dev.config_file

    print("SIS mixers (AD monitors)")
    try:
        sis = read_sis(tm.Devices(path, dev.cfg, dev.m, {"reader"}))
    except Exception as e:
        print(f"  could not be read ({e!r})")
    else:
        print(f"  {'mixer':<9}{'tuned':>9}{'voltage':>11}{'current':>12}")
        for name in tm.BIAS_NAMES:
            v, i = sis[name]
            print(f"  {name} ({ch[name]})  {tuned[name]:5.2f} mV  {v:6.2f} mV  {i:7.1f} uA")

    att = tm.last_loatt()
    if att:
        print("LO Att. (last set from TZ-tools, not read back)")
        ports = dev.m["local_attenuator"]["port"]
        for pol in ["H", "V"]:
            if pol in att:
                print(f"  {pol} (port {ports[pol]})  {att[pol]['mA']:g} mA  at {att[pol]['time'].replace('T', ' ')}")
    else:
        print("LO Att.: no record (not set from TZ-tools in this folder yet)")

    # The generator takes one connection at a time: refused means another
    # program (initialize.py, the observing system) holds it.
    try:
        lo = tm.Devices(path, dev.cfg, dev.m, {"lo1"})
    except ConnectionRefusedError:
        print("1st LO: connection refused (another program is connected to the generator)")
        return
    except Exception as e:
        print(f"1st LO: could not connect ({e!r})")
        return
    try:
        first_lo.report(lo)
    finally:
        lo.disconnect_lo1()


if __name__ == "__main__":
    main()
