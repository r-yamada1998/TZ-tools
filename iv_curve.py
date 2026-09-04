#!/usr/bin/env python3
"""
Measure SIS IV curves.

Leaves the running setup alone. The only config it reads is
device_config_iv.toml, next to this script. Neither the installed
tz_controller nor ~/tz_workspace/device_config.toml is modified.

Usage:
    cd ~/tz_workspace && uv run python iv_curve.py

Output:
    iv_<timestamp>/iv_<mixer>.csv   raw data, one file per mixer
    iv_<timestamp>/iv_curves.png    2x2 IV curves, if matplotlib is available

Sweep range:
    Negative side stops at -5.0 mV. It is there to confirm the origin and goes
    no further: the bias box is meant for positive voltage (advice from the
    receiver developer).

    Positive side stops at 10.0 mV. In the 2026-09-04 sweep the V mixers showed
    their gap at 7.5-7.6 mV, but the H mixers had not turned up by 8.0 mV. With
    three junctions in series the gap should be at 8.4 mV, so the sweep is
    taken well past that. 10.0 mV is the ceiling the existing sis_iv_measure.py
    uses, and equals V_STOP_HARD_LIMIT, above which preflight refuses to run.
"""

import datetime
import logging
import pathlib
import sys
import time
import tomllib

import numpy as np

# tz_controller logs one INFO line per measured point: nearly 600 lines over
# four mixers, which buries the summary. Calling basicConfig first silences it,
# since DeviceBase skips its own call once a handler exists.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from tz_controller.sis_setter import SISSetter  # noqa: E402

# Per-channel conversion factors are required here. Current monitors are
# -500 uA/V and voltage monitors -2.80 mV/V, so one scalar per config section
# cannot scale both correctly.
from tz_controller.sis_reader import SISReader  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CONFIG_FILE = str(HERE / "device_config_iv.toml")

MIXERS = ["v1", "v2", "h1", "h2"]

V_START = -5.0   # mV
V_STOP = 10.0    # mV
V_STEP = 0.1     # mV
SETTLE_TIME = 0.05  # s

# Safety guard
EXPECTED_SETTER_FACTOR = 0.3133333333333333  # = 0.94/3
FACTOR_TOLERANCE = 0.01
V_START_HARD_LIMIT = -5.0   # never sweep below this
V_STOP_HARD_LIMIT = 10.0


def preflight() -> dict:
    """Check the config values before any bias is applied."""
    with open(CONFIG_FILE, "rb") as f:
        cfg = tomllib.load(f)

    setter_cfg = cfg["sis_bias_setter"]
    reader_cfg = cfg["sis_bias_reader"]
    factor = setter_cfg["conversion_factor"]

    rel = abs(factor - EXPECTED_SETTER_FACTOR) / EXPECTED_SETTER_FACTOR
    if rel > FACTOR_TOLERANCE:
        raise SystemExit(
            f"Aborted: setter conversion_factor is {factor}, expected "
            f"{EXPECTED_SETTER_FACTOR}. Sweeping with it would bias the "
            f"junction {factor / EXPECTED_SETTER_FACTOR:.2f}x harder than intended."
        )

    if V_START < V_START_HARD_LIMIT:
        raise SystemExit(
            f"Aborted: V_START={V_START} mV is below the negative hard limit "
            f"{V_START_HARD_LIMIT} mV."
        )

    if V_STOP > V_STOP_HARD_LIMIT:
        raise SystemExit(
            f"Aborted: V_STOP={V_STOP} mV is above the hard limit "
            f"{V_STOP_HARD_LIMIT} mV."
        )

    # With both voltage and current channels registered, the factor has to be
    # a table: a single scalar necessarily mis-scales one of the two.
    if not isinstance(reader_cfg["conversion_factor"], dict):
        raise SystemExit(
            "Aborted: reader conversion_factor is a scalar. Voltage and current "
            "monitors have different monitor gains, so one of them would be "
            "mis-scaled. Use the per-channel table form."
        )

    # Check the registration order. Voltage channels swing 2.86 V, and the last
    # channel in the list leaks into the first one.
    names = list(reader_cfg["channel"].keys())
    if names and names[-1].endswith("_v"):
        print(
            f"Warning: the last registered channel is a voltage monitor "
            f"({names[-1]}). Its residue lands on the first channel "
            f"({names[0]}) at 0.81x. Consider reordering."
        )

    n_points = int(round((V_STOP - V_START) / V_STEP)) + 1
    print("--- measurement settings ---")
    print(f"  config          : {CONFIG_FILE}")
    print(f"  sweep           : {V_START} -> {V_STOP} mV in {V_STEP} mV steps "
          f"({n_points} points)")
    print(f"  negative limit  : {V_START_HARD_LIMIT} mV (never swept below)")
    print(f"  setter factor   : {factor}")
    print(f"  DA output range : {V_START * factor:+.4f} to {V_STOP * factor:+.4f} V "
          f"(range 10V)")
    print(f"  tuned values    : {setter_cfg['tuned']}")
    print(f"  readout order   : {' -> '.join(names)}")
    print(f"  factors         : {reader_cfg['conversion_factor']}")
    print(f"  mixers          : {', '.join(MIXERS)} (one at a time, never together)")
    print()
    if input("Measure with these settings? [yes/no]: ").strip().lower() != "yes":
        raise SystemExit("Aborted. Nothing was written to the DA board.")

    return cfg


def measure(mixer: str) -> dict:
    """Measure the IV curve of one mixer."""
    v_ch = f"{mixer}_v"
    i_ch = mixer

    setter = SISSetter(device_name="sis_bias_setter", config_file=CONFIG_FILE)
    reader = SISReader(device_name="sis_bias_reader", config_file=CONFIG_FILE)

    # Voltage channel first, current channel second: this is the readout order,
    # and reversing it lets the large voltage signal leak into the current one.
    reader.setup(**{v_ch: True, i_ch: True})

    # Trial read, before any bias, to confirm the channel request is accepted.
    reader.run()
    print(f"  trial read at 0 V OK ({v_ch}, {i_ch})")

    f_v = reader.conversion_factors[v_ch]
    f_i = reader.conversion_factors[i_ch]

    rows = []
    try:
        for v_cmd in np.arange(V_START, V_STOP + V_STEP / 2, V_STEP):
            setter.setup(**{mixer: float(v_cmd)})
            setter.run()
            time.sleep(SETTLE_TIME)

            result = reader.run()
            v_mV = result[v_ch]     # converted, mV
            i_uA = result[i_ch]     # converted, uA
            rows.append(
                (float(v_cmd), v_mV / f_v, i_uA / f_i, v_mV, i_uA)
            )
    finally:
        setter.teardown()   # return the channels used to 0 V
        reader.teardown()

    a = np.array(rows)
    return {"cmd_mV": a[:, 0], "v_mon_V": a[:, 1], "i_mon_V": a[:, 2],
            "v_mV": a[:, 3], "i_uA": a[:, 4]}


def save_csv(out_dir: pathlib.Path, mixer: str, d: dict) -> pathlib.Path:
    path = out_dir / f"iv_{mixer}.csv"
    cols = ["cmd_mV", "v_mon_V", "i_mon_V", "v_mV", "i_uA"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(len(d["cmd_mV"])):
            f.write(",".join(f"{d[c][k]}" for c in cols) + "\n")
    return path


def save_plot(out_dir: pathlib.Path, results: dict, tuned: dict) -> None:
    """Give up quietly without matplotlib. The CSV files are already written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable, skipping the plot. The CSV files are written.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, mixer in zip(axes.flatten(), MIXERS):
        d = results[mixer]
        # Keep the sample points. The line is a thin guide between them.
        ax.plot(d["v_mV"], d["i_uA"], lw=0.7, marker=".", ms=3.5, mew=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        if mixer in tuned:
            ax.axvline(tuned[mixer], color="tab:red", ls="--", lw=0.9,
                       label=f"tuned {tuned[mixer]} mV")
            ax.legend(fontsize=8)
        ax.set_title(mixer)
        ax.set_xlabel("Junction voltage [mV]")
        ax.set_ylabel("Current [uA]")
        ax.grid(alpha=0.3)

    fig.suptitle(f"SIS IV curves  {out_dir.name}")
    fig.tight_layout()
    path = out_dir / "iv_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"plot: {path}")


def main() -> None:
    cfg = preflight()
    tuned = dict(cfg["sis_bias_setter"]["tuned"])

    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = HERE / f"iv_{stamp}"
    out_dir.mkdir()

    results = {}
    for mixer in MIXERS:
        print(f"\n=== measuring {mixer} ===")
        d = measure(mixer)
        results[mixer] = d
        print(f"raw data: {save_csv(out_dir, mixer, d)}")

        # Summary on the spot: gap position and leakage current.
        v, i = d["v_mV"], d["i_uA"]
        sub = i[(v > 2.0) & (v < 5.0)]
        print(f"  median subgap current (2-5 mV): {np.median(sub):.1f} uA")
        print(f"  peak current: {i.max():.1f} uA (V={v[int(np.argmax(i))]:.2f} mV)")
        # Take the gap as the first point above five times the leakage current.
        thr = 5 * abs(np.median(sub))
        over = np.where((v > 5.0) & (i > thr))[0]
        if len(over):
            print(f"  gap rise (first point above 5x leakage): {v[over[0]]:.2f} mV")
        else:
            print(f"  no gap rise seen up to {V_STOP} mV")

    save_plot(out_dir, results, tuned)
    print(f"\nDone. Take {out_dir} with you.")
    print("The bias is back at 0 V. Reapply tuned before returning to observation.")


if __name__ == "__main__":
    main()
