#!/usr/bin/env python3
"""
Timing of the receiver chain, without LO and without liquid nitrogen.

Four short tests that decide the waits used by the measurement scripts:

  bias    SIS bias step response. Each mixer is stepped between two sub-gap
          voltages and the voltage and current monitors are read back to back.
          Tells how long the junction takes to follow a bias change.
  sa      Spectrum analyzer: sweep time as reported, time to read one trace,
          and how often the trace content actually changes (one change per
          finished sweep is expected), for the chopper and the IRR settings.
  if      IF switch: the path is switched back and forth between the two
          bands whose noise levels differ most, and traces are read back to
          back. Tells when the new band shows up in the trace.
  soft    Rig.new_sweeps (irr_measure --soft-average) on the real analyzer:
          after an IF switch, are the kept sweeps about one update cycle
          (1.3-1.6 s, measured on 2026-10-22) apart and all at the new band's
          level.

Usage (in the folder that holds this script):
    uv run python timing_test.py [--only bias,sa,if,soft] [--bias-low 2.0] [--bias-high 3.0]

The bias test uses device_config_iv_ros.toml (ROS conversions, reader names);
the other tests use device_config_ros.toml. All biases are 0 at the end.

Output: timing_test_<YYMMDD_HHMMSS>/ with one CSV per test, summary.txt and
bias_step.png.
"""

import argparse
import datetime
import json
import pathlib
import sys
import time

import numpy as np

import rx_performance_common as tm

HERE = pathlib.Path(__file__).resolve().parent
IV_CONFIG = str(HERE / "device_config_iv_ros.toml")
MIXERS = ["v1", "h1", "h2", "v2"]


class Out:
    """Run directory with CSV files and a running summary."""

    def __init__(self) -> None:
        stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.dir = pathlib.Path(".") / f"timing_test_{stamp}"
        self.dir.mkdir()
        self.lines = []

    def csv(self, name: str, header, rows) -> None:
        with (self.dir / name).open("w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    def say(self, text: str) -> None:
        print(text)
        self.lines.append(text)
        (self.dir / "summary.txt").write_text("\n".join(self.lines) + "\n")


# ----------------------------------------------------------------------------
# bias
# ----------------------------------------------------------------------------

def test_bias(out: Out, low: float, high: float, repeat: int, watch: float, dry: bool) -> None:
    out.say(f"== bias step: {low} <-> {high} mV, {repeat} times per mixer, read for {watch} s ==")
    if not (0.0 <= low < high <= 4.0):
        raise SystemExit("Aborted: keep the bias step within 0-4 mV (below the gap).")
    if dry:
        setter = reader = None
    else:
        from tz_controller.sis_setter import SISSetter
        sys.path.insert(0, str(HERE))
        from sis_reader_multi import SISReaderMulti
        setter = SISSetter(device_name="sis_bias_setter", config_file=IV_CONFIG)
        reader = SISReaderMulti(device_name="sis_bias_reader", config_file=IV_CONFIG)

    rows = []
    try:
        for mixer in MIXERS:
            if not dry:
                reader.setup(**{f"{mixer}_v": True, mixer: True})
                setter.setup(**{mixer: low})
                setter.run()
            time.sleep(0 if dry else 2.0)
            step = 0
            for _ in range(repeat):
                for target in (high, low):
                    step += 1
                    if not dry:
                        setter.setup(**{mixer: target})
                        setter.run()
                    t0 = time.monotonic()
                    while True:
                        t = time.monotonic() - t0
                        if t > watch:
                            break
                        if dry:
                            v = target - (target - (low + high - target)) * np.exp(-t / 0.01)
                            i, t = 10 * v, t
                            time.sleep(0.02)
                        else:
                            r = reader.run()
                            v, i = r[f"{mixer}_v"], r[mixer]
                            t = (t + time.monotonic() - t0) / 2   # middle of the read
                        rows.append([mixer, step, target, round(t, 4), round(v, 4), round(i, 3)])
            if not dry:
                setter.setup(**{mixer: 0.0})
                setter.run()
    finally:
        if not dry:
            setter.setup(**{m: 0.0 for m in MIXERS})
            setter.run()
            reader.teardown()
    out.csv("bias_step.csv", ["mixer", "step", "target_mV", "t_s", "v_mV", "i_uA"], rows)

    # Per mixer: time after which the voltage stays within 10 % of the step of its final value
    a = np.array([[MIXERS.index(r[0]), r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    for k, mixer in enumerate(MIXERS):
        times, reads = [], []
        for step in np.unique(a[a[:, 0] == k, 1]):
            s = a[(a[:, 0] == k) & (a[:, 1] == step)]
            t, v = s[:, 3], s[:, 4]
            final = np.median(v[t > watch / 2])
            off = np.abs(v - final) > 0.1 * (high - low)
            idx = np.nonzero(off)[0]
            times.append(0.0 if len(idx) == 0 else (t[idx[-1] + 1] if idx[-1] + 1 < len(t) else np.nan))
            reads.append(np.median(np.diff(t)) if len(t) > 1 else np.nan)
        out.say(f"  {mixer}: settled (within 10 % of the step) after {np.nanmax(times):.3f} s at worst, "
                f"median {np.nanmedian(times):.3f} s; one read every {np.nanmedian(reads) * 1000:.0f} ms")

    plt = _pyplot()
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (k, mixer) in zip(axes.flatten(), enumerate(MIXERS)):
            for step in np.unique(a[a[:, 0] == k, 1]):
                s = a[(a[:, 0] == k) & (a[:, 1] == step)]
                ax.plot(s[:, 3], s[:, 4], ".-", lw=0.6, ms=3, color="C0" if s[0, 2] == high else "C1")
            ax.set_title(mixer)
            ax.set_xlabel("time since bias change [s]")
            ax.set_ylabel("voltage monitor [mV]")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out.dir / "bias_step.png", dpi=110)
        plt.close(fig)


# ----------------------------------------------------------------------------
# sa, if, soft
# ----------------------------------------------------------------------------

def sweep_time(rig) -> float:
    if rig.dry_run:
        return 0.3
    rig.sa.com.open()
    try:
        return float(rig.sa.sa.sweep_time_query())
    finally:
        rig.sa.com.close()


def read_for(rig, seconds: float):
    """Read traces back to back. Returns [(t_mid, read_time, trace), ...]."""
    res = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        a = time.monotonic()
        tr = rig.trace()
        b = time.monotonic()
        res.append(((a + b) / 2 - t0, b - a, tr))
        if rig.dry_run:
            time.sleep(0.05)
    return res


def changes(res):
    """Times at which the trace content changed."""
    return [res[k][0] for k in range(1, len(res)) if not np.array_equal(res[k][2], res[k - 1][2])]


def test_sa(out: Out, rig, watch: float) -> None:
    out.say(f"== spectrum analyzer: read for {watch} s per setting ==")
    rows = []
    for profile in ("sa_chopper", "sa_irr"):
        rig.setup_sa(profile, average_off=True)
        time.sleep(0 if rig.dry_run else 2.0)
        sw = sweep_time(rig)
        res = read_for(rig, watch)
        ch = changes(res)
        gaps = np.diff(ch) if len(ch) > 1 else np.array([np.nan])
        read_s = np.median([r[1] for r in res])
        out.say(f"  {profile}: sweep time {sw:.3f} s, one read {read_s:.3f} s, "
                f"content changed every {np.median(gaps):.3f} s (min {np.min(gaps):.3f}, max {np.max(gaps):.3f})")
        rows += [[profile, round(t, 4), round(rt, 4), round(float(np.mean(tr)), 3)] for t, rt, tr in res]
    out.csv("sa_reads.csv", ["profile", "t_s", "read_s", "mean_dBm"], rows)


def band_levels(rig, seconds: float) -> dict:
    lv = {}
    for band in tm.BANDS:
        rig.open_band(band)
        time.sleep(0 if rig.dry_run else 3.0)
        lv[band] = float(np.mean([np.mean(r[2]) for r in read_for(rig, seconds)]))
    return lv


def test_if(out: Out, rig, repeat: int, watch: float) -> tuple:
    out.say(f"== IF switch: {repeat} switches, read for {watch} s after each ==")
    rig.setup_sa("sa_chopper", average_off=True)
    lv = band_levels(rig, 3.0)
    out.say("  mean level per band [dBm]: " + ", ".join(f"{b} {v:.2f}" for b, v in lv.items()))
    a, b = max(((x, y) for x in lv for y in lv if x < y), key=lambda p: abs(lv[p[0]] - lv[p[1]]))
    diff = abs(lv[a] - lv[b])
    out.say(f"  using {a} and {b}, {diff:.2f} dB apart")
    if diff < 0.3:
        out.say("  difference too small to time the switch; skipped. If all four bands agree within"
                " about 0.1 dB, the IF switch may not be switching at all.")
        return a, b, lv
    rows, lags = [], []
    rig.open_band(a)
    time.sleep(0 if rig.dry_run else 3.0)
    for k in range(repeat):
        target = b if k % 2 == 0 else a
        rig.open_band(target)
        res = read_for(rig, watch)
        mid = (lv[a] + lv[b]) / 2
        on_new = [(t, np.mean(tr)) for t, _, tr in res if (np.mean(tr) - mid) * (lv[target] - mid) > 0]
        lags.append(on_new[0][0] if on_new else np.nan)
        rows += [[k + 1, target, round(t, 4), round(float(np.mean(tr)), 3)] for t, _, tr in res]
    out.csv("if_switch.csv", ["switch", "to", "t_s", "mean_dBm"], rows)
    out.say(f"  new band first seen after {np.nanmedian(lags):.2f} s (median), "
            f"{np.nanmax(lags):.2f} s at worst, over {repeat} switches")
    return a, b, lv


def test_soft(out: Out, rig, bands: tuple, levels: dict, repeat: int, n: int) -> None:
    out.say(f"== soft average: {n} sweeps after each of {repeat} IF switches (IRR settings) ==")
    rig.setup_sa("sa_irr", average_off=True)
    time.sleep(0 if rig.dry_run else 2.0)
    sw = sweep_time(rig)
    a, b = bands
    rows = []
    for k in range(repeat):
        target = b if k % 2 == 0 else a
        other = a if target == b else b
        rig.open_band(target)
        sweeps = rig.new_sweeps(n)
        times = [t for t, _ in sweeps]
        means = [float(np.mean(tr)) for _, tr in sweeps]
        if abs(levels[target] - levels[other]) < 0.3:
            nearer = ["?"] * len(means)   # bands too alike to tell
        else:
            nearer = ["new" if abs(m - levels[target]) < abs(m - levels[other]) else "OLD" for m in means]
        out.say(f"  to {target}: sweeps at {', '.join(f'{t:.1f}' for t in times)} s; "
                f"level {', '.join(f'{m:.2f}' for m in means)} dBm ({', '.join(nearer)})")
        rows += [[k + 1, target, j, t, round(m, 3)] for j, (t, m) in enumerate(zip(times, means))]
    out.csv("soft_average.csv", ["switch", "to", "sweep", "t_s", "mean_dBm"], rows)
    out.say(f"  sweep time reported {sw:.2f} s. Expected: sweeps 1.3-1.6 s apart (the update cycle), all 'new'. "
            "Levels are from the chopper settings run, so the IRR span may shift them a little.")


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--only", default="bias,sa,if,soft", help="comma separated: bias,sa,if,soft")
    p.add_argument("--bias-low", type=float, default=2.0, help="[mV]")
    p.add_argument("--bias-high", type=float, default=3.0, help="[mV]")
    p.add_argument("--repeat", type=int, default=4, help="bias steps per mixer and IF switches")
    tm.add_common_args(p)
    args = p.parse_args()
    tests = set(args.only.split(","))
    unknown = tests - {"bias", "sa", "if", "soft"}
    if unknown:
        raise SystemExit(f"Unknown test: {', '.join(sorted(unknown))}")

    tm.AUTO_YES = args.yes
    out = Out()
    out.say(f"timing_test {datetime.datetime.now().isoformat()}  tests: {args.only}  dry run: {args.dry_run}")
    (out.dir / "arguments.json").write_text(json.dumps(vars(args), indent=2))
    tm.ask_yes("LO off is fine. SIS bias switch on OPERATE, analyzer and IF switch on. Start")

    if "bias" in tests:
        test_bias(out, args.bias_low, args.bias_high, args.repeat, 1.5, args.dry_run)

    if tests & {"sa", "if", "soft"}:
        rig = tm.start(args, use={"sa", "switch"})
        if "sa" in tests:
            test_sa(out, rig, 15.0)
        bands, levels = ("HU", "VU"), None
        if "if" in tests or "soft" in tests:
            a, b, levels = test_if(out, rig, args.repeat, 10.0) if "if" in tests else (None, None, None)
            if levels is None:
                rig.setup_sa("sa_chopper", average_off=True)
                levels = band_levels(rig, 3.0)
                a, b = max(((x, y) for x in levels for y in levels if x < y),
                           key=lambda q: abs(levels[q[0]] - levels[q[1]]))
            bands = (a, b)
        if "soft" in tests:
            test_soft(out, rig, bands, levels, args.repeat, 3)

    out.say(f"done: {out.dir}")


if __name__ == "__main__":
    main()
