#!/usr/bin/env python3
"""
Measure the SIS IV curves of the four mixers, one mixer at a time.

Usage (in the folder that holds this script):
    uv run python iv_curve_measure.py              junction voltage -4.9 to +10 mV
    uv run python iv_curve_measure.py --probe      which AD channel moves with which mixer
    uv run python iv_curve_measure.py --replot DIR draw the figures again from the CSV files

Output:
    iv_<timestamp>/iv_<mixer>.csv      per mixer: setting, monitors as read [V],
                                       junction voltage [mV] and current [uA]
    iv_<timestamp>/iv_curves.png       2x2 IV curves (if matplotlib is there)

--probe sweeps one mixer at a time from 0 to 8 mV (positive only) and records
all 16 AD channels as raw volts ([ad_all_channels] in measure_config.toml).
The channels that move show the wiring: linear in the setting = voltage
monitor, rising at the gap = current monitor. Use it after work on the
receiver wiring.
    probe_<timestamp>/probe_<mixer>.csv, probe_channels.png, probe_iv.png

--replot reads iv_*.csv or probe_*.csv in DIR and writes the PNG files next
to them, without touching the receiver.

Sweep range ([iv] in measure_config.toml): the junction must never see less
than -5 mV or more than +10 mV (receiver developer). The sweep starts at
-4.9 mV because a setting of -4.9 mV read -5.02 to -5.05 mV (2026-10-22).
All biases are 0 V at the end.
"""

import argparse
import datetime
import pathlib
import time

import numpy as np

import rx_performance_common as tm

MIXERS = ["v1", "v2", "h1", "h2"]

# Monitor channels of each mixer in the probe CSV (odd = current, even = voltage)
PROBE_MAP = {"v1": ("c9", "c10"), "h1": ("c11", "c12"),
             "h2": ("c13", "c14"), "v2": ("c15", "c16")}


def sweep_points(r: dict) -> np.ndarray:
    return np.arange(r["start"], r["stop"] + r["step"] / 2, r["step"])


def preflight(dev, probe: bool) -> None:
    """Check the sweep before anything is sent to the DA board, then ask."""
    iv = dev.m["iv"]
    r = iv["probe"] if probe else iv
    if r["start"] < iv["hard_min"] or r["stop"] > iv["hard_max"]:
        raise SystemExit(f"Aborted: the sweep {r['start']} to {r['stop']} mV is outside "
                         f"{iv['hard_min']} to {iv['hard_max']} mV.")
    if not probe:
        names = list(dev.cfg["sis_bias_reader"]["channel"])
        missing = [n for m in MIXERS for n in (f"{m}_v", m) if n not in names]
        if missing:
            raise SystemExit(f"Aborted: sis_bias_reader.channel has no {missing} "
                             f"in {dev.config_file}.")
        # The last registered channel leaks into the first one; the voltage
        # channels swing 2 V, so they must not be last.
        if names[-1].endswith("_v"):
            print(f"Warning: a voltage channel ({names[-1]}) is registered last; it leaks into {names[0]}.")
    n = len(sweep_points(r))
    what = "all 16 AD channels as raw volts" if probe else "voltage and current monitors"
    print(f"Sweep {r['start']} to {r['stop']} mV, {r['step']} mV steps ({n} points), "
          f"mixers {', '.join(MIXERS)} one at a time; {what}")
    print(f"Never below {iv['hard_min']} mV. All biases are 0 V at the end.")
    if tm.AUTO_YES:
        return
    if input("start [y/n]: ").strip().lower() != "y":
        raise SystemExit("Aborted. Nothing was sent to the DA board.")


def open_boards(dev):
    """SISSetter and SISReader, set up anew for each mixer (their __init__
    opens and initializes the DA and AD boards each time)."""
    return tm.Devices(dev.config_file, dev.cfg, dev.m, {"sis", "reader"})


def measure(dev, mixer: str) -> dict:
    """IV curve of one mixer."""
    dev = open_boards(dev)
    iv = dev.m["iv"]
    v_ch, i_ch = f"{mixer}_v", mixer

    # Voltage channel first, current second: this is the reading order.
    # The other way round the current channel picks up the voltage.
    extra = dev.setup_monitors([v_ch, i_ch])
    dev.read_monitors(extra)   # trial reading at 0 V: the channel names are valid
    f = dev.monitor_factors()
    f_v, f_i = f[v_ch], f[i_ch]

    rows = []
    try:
        for v_set in sweep_points(iv):
            dev.sis.setup(**{mixer: float(v_set)})
            dev.sis.run()
            time.sleep(iv["settle"])

            result = dev.read_monitors(extra)
            v_mV = result[v_ch]
            i_uA = result[i_ch]
            rows.append((float(v_set), v_mV / f_v, i_uA / f_i, v_mV, i_uA))
    finally:
        dev.sis.teardown()   # the channel back to 0 V
        dev.reader.teardown()

    a = np.array(rows)
    return {"cmd_mV": a[:, 0], "v_mon_V": a[:, 1], "i_mon_V": a[:, 2],
            "v_mV": a[:, 3], "i_uA": a[:, 4]}


def probe(dev, mixer: str) -> dict:
    """Sweep one mixer and record the raw voltage of every AD channel."""
    from tz_controller.sis_reader import SISReader
    dev = tm.Devices(dev.config_file, dev.cfg, dev.m, {"sis"})
    ad = SISReader(device_name="ad_all_channels", config_file=str(tm.MEASURE_CONFIG))
    r = dev.m["iv"]["probe"]
    ad.setup()   # every channel of [ad_all_channels]
    names = list(ad.selected_names)
    ad.run()     # trial reading at 0 V
    cols = {"mv": []}
    cols.update({n: [] for n in names})
    try:
        for v_set in sweep_points(r):
            dev.sis.setup(**{mixer: float(v_set)})
            dev.sis.run()
            time.sleep(dev.m["iv"]["settle"])
            result = ad.run()
            cols["mv"].append(float(v_set))
            for n in names:
                cols[n].append(result[n])
    finally:
        dev.sis.teardown()
        ad.teardown()
    return {k: np.array(v) for k, v in cols.items()}


def save_csv(path: pathlib.Path, d: dict) -> None:
    cols = list(d)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(len(d[cols[0]])):
            f.write(",".join(f"{d[c][k]}" for c in cols) + "\n")


def read_csv(path: pathlib.Path) -> dict:
    a = np.genfromtxt(path, delimiter=",", names=True)
    return {n: a[n] for n in a.dtype.names}


def summary(mixer: str, d: dict) -> str:
    """One line: subgap current and the gap (steepest rise above 5 mV)."""
    v, i = d["v_mV"], d["i_uA"]
    sub = float(np.median(i[(v > 2.0) & (v < 5.0)]))
    k = np.where(v > 5.0)[0]
    slope = np.gradient(i[k], v[k])
    gap = f"gap {v[k][int(np.argmax(slope))]:.2f} mV" if len(k) > 2 else "gap not reached"
    return f"{mixer}: {gap}, subgap (2-5 mV) {sub:.1f} uA, max {i.max():.1f} uA"


def probe_report(d: dict) -> None:
    """Print how every channel responded, with a verdict as an aid."""
    x = d["mv"]
    names = [n for n in d if n != "mv"]
    stats = {}
    for n in names:
        y = d[n]
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (slope * x + icpt)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        stats[n] = {"span": float(y.max() - y.min()), "slope": float(slope),
                    "r2": 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0,
                    "linear_span": abs(float(slope)) * float(x.max() - x.min())}
    # Most columns are unrelated, so the median swing is the noise.
    noise = float(np.median([s["span"] for s in stats.values()]))
    threshold = max(10.0 * noise, 0.05)

    def verdict(s):
        # R^2 first: noise reached R^2 = 0.14 at most, so a strong correlation
        # is real even with a small swing (a few AD counts, 1/2048 V each).
        if s["r2"] > 0.9 and s["span"] > 4 * (1 / 2048):
            return "voltage monitor (linear in the setting)"
        if s["span"] < threshold:
            return "flat"
        if s["linear_span"] < 0.2 * s["span"]:
            return "check this (large but uncorrelated with the sweep)"
        return "current monitor (rises at the gap)"

    print(f"  noise floor (median swing) {noise:.5f} V, threshold {threshold:.5f} V")
    print(f"  {'ch':<5}{'swing[V]':>12}{'slope[V/mV]':>13}{'R^2':>9}  verdict")
    for n in names:
        s = stats[n]
        print(f"  {n:<5}{s['span']:>12.5f}{s['slope']:>13.5f}{s['r2']:>9.4f}  {verdict(s)}")
    hits = [n for n in names if verdict(stats[n]) != "flat"]
    print(f"  --> responded: {', '.join(hits)}" if hits else "  --> nothing responded")


# ----------------------------------------------------------------------------
# Figures (skipped quietly without matplotlib; the CSV files are written first)
# ----------------------------------------------------------------------------

def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_iv(out_dir: pathlib.Path, results: dict, tuned: dict, xlim: tuple, name="iv_curves.png", title="SIS IV curves") -> None:
    plt = _pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, mixer in zip(axes.flatten(), MIXERS):
        if mixer not in results:
            ax.axis("off")
            continue
        d = results[mixer]
        # Keep the sample points; the line only guides the eye.
        ax.plot(d["v_mV"], d["i_uA"], lw=0.7, marker=".", ms=3.5, mew=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        if mixer in tuned:
            ax.axvline(tuned[mixer], color="tab:red", ls="--", lw=0.9, label=f"tuned {tuned[mixer]} mV")
            ax.legend(fontsize=8)
        ax.set_xlim(*xlim)
        ax.set_title(mixer)
        ax.set_xlabel("Junction voltage [mV]")
        ax.set_ylabel("Current [uA]")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}  {out_dir.name}")
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=140)
    plt.close(fig)


def plot_probe(out_dir: pathlib.Path, results: dict) -> None:
    """Every AD channel against the setting, the moving ones in colour."""
    plt = _pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, mixer in zip(axes.flatten(), MIXERS):
        if mixer not in results:
            ax.axis("off")
            continue
        d = results[mixer]
        for n in [k for k in d if k != "mv"]:
            span = float(d[n].max() - d[n].min())
            if span < 0.05:
                ax.plot(d["mv"], d[n], color="0.8", lw=0.5, marker=".", ms=2, mew=0, zorder=1)
            else:
                ax.plot(d["mv"], d[n], lw=0.7, marker=".", ms=3.5, mew=0, zorder=2,
                        label=f"{n} ({span:.2f} V)")
        ax.set_title(f"{mixer} sweep")
        ax.set_xlabel("Bias setting [mV]")
        ax.set_ylabel("Raw AD voltage [V]")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8)
    fig.suptitle(f"AD channel probe  {out_dir.name}")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_channels.png", dpi=140)
    plt.close(fig)


def probe_as_iv(results: dict, factors: dict) -> dict:
    """Probe columns converted with the channel map and the monitor factors."""
    out = {}
    for mixer, d in results.items():
        i_col, v_col = PROBE_MAP[mixer]
        if i_col in d and v_col in d:
            out[mixer] = {"v_mV": d[v_col] * factors[f"{mixer}_v"], "i_uA": d[i_col] * factors[mixer]}
    return out


def replot(dir_: pathlib.Path, meas: dict, cfg: dict) -> None:
    iv = meas["iv"]
    xlim = (iv["hard_min"] - 1.0, iv["hard_max"] + 1.0)
    tuned = dict(cfg["sis_bias_setter"]["tuned"])
    ivs = {m: read_csv(dir_ / f"iv_{m}.csv") for m in MIXERS if (dir_ / f"iv_{m}.csv").exists()}
    probes = {m: read_csv(dir_ / f"probe_{m}.csv") for m in MIXERS if (dir_ / f"probe_{m}.csv").exists()}
    if not ivs and not probes:
        raise SystemExit(f"No iv_*.csv or probe_*.csv in {dir_}")
    if ivs:
        plot_iv(dir_, ivs, tuned, xlim)
    if probes:
        f = cfg["sis_bias_reader"]["conversion_factor"]
        factors = {n: float(f[n] if isinstance(f, dict) else f) for n in cfg["sis_bias_reader"]["channel"]}
        plot_probe(dir_, probes)
        plot_iv(dir_, probe_as_iv(probes, factors), tuned, xlim, "probe_iv.png", "SIS IV curves from the probe")
    print(f"Figures written in {dir_}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--probe", action="store_true",
                   help="record all 16 AD channels as raw volts, 0 to 8 mV")
    p.add_argument("--replot", metavar="DIR", help="only draw the figures from the CSV files in DIR")
    tm.add_common_args(p)
    args = p.parse_args()

    if args.replot:
        _, cfg, meas = tm.load_configs(args.device_config)
        replot(pathlib.Path(args.replot), meas, cfg)
        return

    dev = tm.start(args, use=set())
    preflight(dev, args.probe)
    iv = dev.m["iv"]
    xlim = (iv["hard_min"] - 1.0, iv["hard_max"] + 1.0)
    tuned = dict(dev.cfg["sis_bias_setter"]["tuned"])
    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    if args.probe:
        out_dir = tm.HERE / f"probe_{stamp}"
        out_dir.mkdir()
        results = {}
        for mixer in MIXERS:
            print(f"\n=== sweeping {mixer} ===")
            results[mixer] = probe(dev, mixer)
            save_csv(out_dir / f"probe_{mixer}.csv", results[mixer])
            probe_report(results[mixer])
        replot(out_dir, dev.m, dev.cfg)
        print(f'Raw data: "{out_dir.name}". All biases are 0 V; set the operating values again before observing.')
        return

    out_dir = tm.HERE / f"iv_{stamp}"
    out_dir.mkdir()
    results = {}
    for mixer in MIXERS:
        d = measure(dev, mixer)
        results[mixer] = d
        save_csv(out_dir / f"iv_{mixer}.csv", d)
        print(summary(mixer, d))

    plot_iv(out_dir, results, tuned, xlim)
    print(f'Raw data: "{out_dir.name}". All biases are 0 V; set the operating values again before observing.')


if __name__ == "__main__":
    main()
