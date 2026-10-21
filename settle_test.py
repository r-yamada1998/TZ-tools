#!/usr/bin/env python3
"""
Measure how long the chopper Trx takes to settle after an SIS bias change.

trx_map.py waits 5.8 s after each bias2 step, as the ROS script did. The ROS
logs show that this wait was set by the spectrum analyzer, not by the SIS:
under ROS a new trace arrived only every ~2.5 s, and the first trace after a
change still showed the previous point. Whether the junction itself settles
in well under a second could not be told from the logs. This test measures it
with the new system, so that trx_map.py --settle can be shortened safely.

Not a ROS port. The chopper must be rotating and the LO on, as for trx_map.

Usage (in the folder that holds this script):
    uv run python settle_test.py HU \
        --a 6.4 7.3 7.3 6.3 --b 6.4 7.3 7.8 6.3 [--repeat 5] [--watch 10]

    --a V1 H1 H2 V2   the point the receiver is at now; restored at the end
    --b V1 H1 H2 V2   a nearby point with clearly different Trx in the band

Sequence: set A and wait 10 s. Then, --repeat times, go to B and to A; after
each change read traces back to back for --watch seconds, recording when each
read started and ended relative to the change. The bias is left at A.

Result (rx_performance_analysis.py): Trx against time since the change, and
per change the time after which every trace stays within 5 % of the final
value. A trace read right after a change can still hold a sweep that began
before it; compare the settling time with the sweep time in meta.json.
"""

import argparse
import time

import rx_performance_common as tm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("band", choices=tm.BANDS)
    p.add_argument("--a", nargs=4, type=float, required=True, metavar=tuple(tm.BIAS_NAMES))
    p.add_argument("--b", nargs=4, type=float, required=True, metavar=tuple(tm.BIAS_NAMES))
    p.add_argument("--repeat", type=int, default=5, help="number of A -> B -> A cycles")
    p.add_argument("--watch", type=float, default=10.0, help="read traces for this long after each change [s]")
    tm.add_common_args(p)
    args = p.parse_args()

    points = {"A": dict(zip(tm.BIAS_NAMES, args.a)), "B": dict(zip(tm.BIAS_NAMES, args.b))}
    n_changes = 2 * args.repeat
    print(f"{n_changes} changes x {args.watch:.0f} s, about {(n_changes * args.watch + 10) / 60:.0f} min")

    rig = tm.start(args, use={"sis", "sa", "switch"})
    for name, values in points.items():
        for k, v in values.items():
            if not rig.m["bias_min"] <= v <= rig.m["bias_max"]:
                raise SystemExit(f"Aborted: {name} {k} = {v} mV is outside the bias limits.")
    if not args.keep_sa:
        rig.setup_sa("sa_chopper")

    rec = tm.Recorder("settle_test", {
        **tm.base_meta(rig, args),
        "band": args.band,
        "point_a": points["A"],
        "point_b": points["B"],
        "sa_params": rig.sa_params(),
    }, columns=["change", "target", "t_read_start_s", "t_read_end_s"])

    rig.open_band(args.band)
    tm.ask_yes("Chopper rotating and LO on. Start")
    rig.set_bias(**points["A"])
    tm.sleep(rig, 10.0)
    try:
        change = 0
        for _ in range(args.repeat):
            for target in ["B", "A"]:
                change += 1
                rig.set_bias(**points[target])
                t0 = time.monotonic()
                n = 0
                while True:
                    start = time.monotonic() - t0
                    if start > args.watch:
                        break
                    trace = rig.trace()
                    end = time.monotonic() - t0
                    rec.add(trace, change=change, target=target,
                            t_read_start_s=round(start, 3), t_read_end_s=round(end, 3))
                    n += 1
                    if rig.dry_run and n >= 20:
                        break
                print(f"change {change} -> {target}: {n} traces")
    finally:
        rig.set_bias(**points["A"])
        print("SIS bias left at A: " + ", ".join(f"{k} = {v}" for k, v in points["A"].items()))

    tm.finish(rec)


if __name__ == "__main__":
    main()
