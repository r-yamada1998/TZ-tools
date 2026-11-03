#!/usr/bin/env python3
"""
Put TZ at the operating point for an observation.

It switches the 1st LO on, sets the LO Att. and the SIS biases, waits for
"finished", then undoes all of it.

The operating values are read from the device config, as initialize.py does:
    SIS biases       sis_bias_setter.tuned
    LO Att. [mA]     local_attenuator.channels
    1st LO [GHz]     signal_generator.defaults.freq

Usage (in the folder that holds this script):
    uv run python initialize.py                      the operating values above
    uv run python initialize.py --loatt 15 15        other LO Att. currents [mA] for this run
    uv run python initialize.py --lo1 15.8           other 1st LO frequency [GHz] for this run
    uv run python initialize.py --no-lo1             leave the 1st LO alone

Order (the LO Att. is at 70 70 mA, the LO weak, while the 1st LO is
switched; receiver developer):
    start:    LO Att. 70 70 -> 1st LO on -> LO Att. operating values -> SIS biases
    finished: SIS biases 0 -> LO Att. 70 70 -> 1st LO off
Ctrl+C also runs the finished steps. If the terminal is closed instead, the
values stay. The 1st LO connection is closed while waiting (the generator
takes one connection at a time), so tz_status.py can read the state from
another terminal.
"""

import argparse

import first_lo
import rx_performance_common as tm


def wait_for_finished() -> None:
    """Wait until "finished" is typed, as initialize.py does."""
    if tm.AUTO_YES:
        return
    print('\nWhen your observation finished, please type "finished"')
    while True:
        user_input = input("> ").strip().lower()
        if user_input == "finished":
            return
        print('Please type "finished" when your observation is done.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--loatt", nargs=2, type=float, default=None, metavar=("H", "V"),
                   help="LO Att. currents [mA] (default: local_attenuator.channels)")
    p.add_argument("--lo1", type=float, default=None,
                   help="1st LO frequency [GHz] (default: signal_generator.defaults.freq)")
    p.add_argument("--no-lo1", action="store_true", help="do not switch the 1st LO")
    tm.add_common_args(p)
    args = p.parse_args()

    _, cfg, meas = tm.load_configs(args.device_config)
    if args.loatt is None:
        att = tm.tuned_loatt(cfg, meas)
        args.loatt = [att["H"], att["V"]]
    if args.lo1 is None:
        args.lo1 = tm.tuned_lo1_ghz(cfg)
    weak = meas["local_attenuator"]["weak"]

    for value in args.loatt:
        # TZ manual 3.3: 0-100 mA
        if not 0.0 <= value <= 100.0:
            raise SystemExit(f"Aborted: {value} mA is outside 0-100 mA.")
    if not args.no_lo1 and not 0.0 < args.lo1 <= 20.0:
        raise SystemExit(f"Aborted: {args.lo1} GHz is outside 0-20 GHz.")
    dev = tm.start(args, use={"sis", "loatt"} if args.no_lo1 else {"sis", "loatt", "lo1"})
    bias = tm.tuned_bias(dev.cfg)
    ch = dev.cfg["sis_bias_setter"]["channel"]
    ports = dev.m["local_attenuator"]["port"]

    print("Starting instrument setup...")
    if not args.no_lo1:
        dev.set_loatt("H", weak)
        dev.set_loatt("V", weak)
        dev.lo1_on(args.lo1)
        if not first_lo.report_after_on(dev, args.lo1):
            # Kept going so that the setup is complete and "finished" undoes it;
            # the observer must not observe until the 1st LO is fixed.
            print("WARNING: the 1st LO is not right (NG above). Do not observe; "
                  "check it with: uv run python tz_status.py")
        # Free the generator while waiting: it takes one connection at a time,
        # and the observing system or tz_status.py may need it.
        dev.disconnect_lo1()
    dev.set_loatt("H", args.loatt[0])
    dev.set_loatt("V", args.loatt[1])
    print(f"LO attenuator: H (port {ports['H']}) = {args.loatt[0]} mA, "
          f"V (port {ports['V']}) = {args.loatt[1]} mA")
    dev.set_bias(**bias)
    print("SIS bias: " + ", ".join(f"{k} ({ch[k]}) = {v} mV" for k, v in bias.items()))
    print("All instruments have been started successfully.")

    try:
        wait_for_finished()
        print("Observation finished. Starting teardown...")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Starting teardown...")
    finally:
        try:
            dev.zero_bias()
            print("SIS bias: all 0 mV")
        except Exception as e:
            print(f"SIS teardown failed: {e}")
        try:
            dev.set_loatt("H", weak)
            dev.set_loatt("V", weak)
            print(f"LO attenuator: H = V = {weak} mA")
        except Exception as e:
            print(f"LoAttenuator teardown failed: {e}")
        if not args.no_lo1:
            try:
                dev.reconnect_lo1()
                dev.lo1_off()
                first_lo.check_off(first_lo.report(dev))
            except Exception as e:
                print(f"1st LO teardown failed: {e!r}")
                print("Turn it off later with: uv run python first_lo.py off")
            try:
                dev.disconnect_lo1()
            except Exception:
                pass
        print("Teardown completed.")


if __name__ == "__main__":
    main()
