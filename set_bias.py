#!/usr/bin/env python3
"""
Set the four SIS biases.

Usage (in the folder that holds this script):
    uv run python set_bias.py V1 H1 H2 V2          e.g. 6.1 7.2 7.4 6.1 [mV]
    uv run python set_bias.py --tuned              the operating values (sis_bias_setter.tuned)

The values stay after the script ends. Limits: [bias] in measure_config.toml (0-9 mV).
"""

import argparse

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("values", nargs="*", type=float, metavar="MV", help="V1 H1 H2 V2 [mV]")
    p.add_argument("--tuned", action="store_true", help="use sis_bias_setter.tuned")
    tm.add_common_args(p)
    args = p.parse_args()

    dev = tm.start(args, use={"sis"})
    if args.tuned:
        bias = tm.tuned_bias(dev.cfg)
    elif len(args.values) == 4:
        bias = dict(zip(tm.BIAS_NAMES, args.values))
    else:
        raise SystemExit("Give V1 H1 H2 V2, or --tuned.")
    dev.set_bias(**bias)
    ch = dev.cfg["sis_bias_setter"]["channel"]
    print("SIS bias set: " + ", ".join(f"{k} ({ch[k]}) = {v} mV" for k, v in bias.items()))


if __name__ == "__main__":
    main()
