#!/usr/bin/env python3
"""
Identify which AD channel carries which mixer's voltage and current monitor.

Method:
    Sweep one mixer at a time from 0 to 8 mV while recording every AD channel
    as raw volts. Only the channels wired to the swept mixer respond, so the
    wiring follows from which column moves.

Reading the result:
    rises linearly with the command  -> voltage monitor of that mixer
    turns up sharply near the gap    -> current monitor of that mixer
    stays flat                       -> unrelated or not connected

Usage:
    uv run python probe_channels.py

Safety:
    The sweep is positive only, 0 to 8 mV, stopping just past the gap at about
    7.5 mV. It never goes negative: the bias box is meant for positive voltage
    (advice from the receiver developer).
"""

import pathlib
import sys
import time
import tomllib

import numpy as np

from tz_controller.sis_setter import SISSetter
from tz_controller.sis_reader import SISReader

# Anchor everything to the script's own directory, so the same config is read
# and the csv files land in the same place regardless of the working directory.
HERE = pathlib.Path(__file__).resolve().parent
CONFIG_FILE = str(HERE / "probe_config.toml")

MIXERS = ["v1", "v2", "h1", "h2"]

V_START = 0.0    # mV
V_STOP = 8.0     # mV, just past the gap at about 7.5 mV
V_STEP = 0.2     # mV
SETTLE_TIME = 0.05  # s

# Safety guard.
# Setter factor of the production device_config.toml: a command of 8 mV puts
# 2.507 V on the DA board. Picking up the 1.0 that sits in some test configs
# would put 8 V there instead, about 25.5 mV on the mixer, 3.2x the intent.
EXPECTED_SETTER_FACTOR = 0.3133333333333333
FACTOR_TOLERANCE = 0.01   # relative
V_STOP_HARD_LIMIT = 10.0  # mV, the ceiling the existing sis_iv_measure.py uses


def preflight() -> float:
    """
    Check the config values before any bias is applied.

    Raises on anything wrong, with nothing written to the DA board.
    Returns the setter conversion factor.
    """
    with open(CONFIG_FILE, "rb") as f:
        cfg = tomllib.load(f)

    factor = cfg["sis_bias_setter"]["conversion_factor"]
    rel = abs(factor - EXPECTED_SETTER_FACTOR) / EXPECTED_SETTER_FACTOR
    if rel > FACTOR_TOLERANCE:
        raise SystemExit(
            f"Aborted: setter conversion_factor is {factor}, expected "
            f"{EXPECTED_SETTER_FACTOR} as in the production device_config.toml. "
            f"Sweeping with it would drive the DA board to "
            f"{V_STOP * factor:.3f} V, biasing the junction "
            f"{factor / EXPECTED_SETTER_FACTOR:.1f}x harder than intended."
        )

    if V_STOP > V_STOP_HARD_LIMIT:
        raise SystemExit(
            f"Aborted: V_STOP={V_STOP} mV is above the hard limit "
            f"{V_STOP_HARD_LIMIT} mV."
        )

    reader_factor = cfg["sis_bias_reader"]["conversion_factor"]
    if reader_factor != 1.0:
        raise SystemExit(
            f"Aborted: reader conversion_factor is {reader_factor}. Channel "
            "identification needs 1.0, so that columns are compared as raw volts."
        )

    print("--- sweep settings ---")
    print(f"  config         : {CONFIG_FILE}")
    print(f"  sweep          : {V_START} -> {V_STOP} mV in {V_STEP} mV steps")
    print(f"  setter factor  : {factor}")
    print(f"  max DA output  : {V_STOP * factor:.4f} V (range 10V)")
    print(f"  tuned values   : {cfg['sis_bias_setter']['tuned']}")
    print(f"  mixers         : {', '.join(MIXERS)} (one at a time, never together)")
    print()
    answer = input("Sweep with these settings? [yes/no]: ").strip().lower()
    if answer != "yes":
        raise SystemExit("Aborted. Nothing was written to the DA board.")

    return factor


def probe(mixer: str) -> tuple[list[float], dict[str, list[float]]]:
    """Sweep one mixer and record the raw voltage of every AD channel."""
    setter = SISSetter(device_name="sis_bias_setter", config_file=CONFIG_FILE)
    reader = SISReader(device_name="sis_bias_reader", config_file=CONFIG_FILE)

    reader.setup()  # no arguments = every channel registered in config
    names = list(reader.selected_names)

    # Trial read, before any bias is applied, to confirm the channel request is
    # accepted. An invalid channel raises here, with no voltage on the mixer.
    reader.run()
    print(f"  trial read at 0 V OK ({len(names)} ch)")

    commands: list[float] = []
    columns: dict[str, list[float]] = {name: [] for name in names}

    try:
        for v_cmd in np.arange(V_START, V_STOP + V_STEP / 2, V_STEP):
            setter.setup(**{mixer: float(v_cmd)})
            setter.run()
            time.sleep(SETTLE_TIME)

            result = reader.run()
            commands.append(float(v_cmd))
            for name in names:
                columns[name].append(result[name])
    finally:
        setter.teardown()
        reader.teardown()

    return commands, columns


def stats_of(commands: list[float], values: list[float]) -> dict:
    """Summarise how one column responded to the sweep."""
    x = np.asarray(commands, dtype=float)
    y = np.asarray(values, dtype=float)

    span = float(y.max() - y.min())
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0

    return {
        "span": span,
        "slope": float(slope),
        "r2": r2,
        # Swing of the linear component alone. Small against span means the
        # movement did not come from the sweep.
        "linear_span": abs(float(slope)) * float(x.max() - x.min()),
    }


def classify(st: dict, threshold: float) -> str:
    """
    Decide what a column is, against a threshold taken from the noise floor.

    R^2 is tested first. Measured noise reaches R^2 = 0.14 at most, so a column
    that correlates strongly with the sweep is real even when its swing is
    small. That keeps a low-gain voltage monitor from being missed.
    """
    # AD resolution is 1/2048 V. A few counts of movement is the minimum for
    # a correlation to mean anything.
    if st["r2"] > 0.9 and st["span"] > 4 * (1 / 2048):
        return "*** voltage monitor (linear in the command) ***"

    if st["span"] < threshold:
        return "flat"

    if st["linear_span"] < 0.2 * st["span"]:
        # Moves a lot but not with the sweep. Suspect a bad contact or pickup.
        return "check this (large but uncorrelated with the sweep)"

    return "*** current monitor (rises non-linearly with the sweep) ***"


def main() -> None:
    preflight()

    for mixer in MIXERS:
        print(f"\n=== sweeping {mixer} ===")
        commands, columns = probe(mixer)

        out = str(HERE / f"probe_{mixer}.csv")
        names = list(columns.keys())
        with open(out, "w") as f:
            f.write("mv," + ",".join(names) + "\n")
            for i, mv in enumerate(commands):
                f.write(f"{mv}," + ",".join(f"{columns[n][i]}" for n in names) + "\n")
        print(f"raw data: {out}")

        stats = {name: stats_of(commands, columns[name]) for name in names}

        # Take the noise floor from the data. Most columns are unrelated, so the
        # median swing is the size of the noise. Ten times that is the threshold.
        noise = float(np.median([st["span"] for st in stats.values()]))
        threshold = max(10.0 * noise, 0.05)
        print(f"  noise floor (median swing)={noise:.5f} V  threshold={threshold:.5f} V")
        print(f"  {'ch':<5}{'swing[V]':>12}{'slope[V/mV]':>13}{'R^2':>9}  verdict")

        # Always print every column. The verdict is an aid; the numbers are
        # there so the result can be judged by eye.
        for name in names:
            st = stats[name]
            print(
                f"  {name:<5}{st['span']:>12.5f}{st['slope']:>13.5f}"
                f"{st['r2']:>9.4f}  {classify(st, threshold)}"
            )

        hits = [n for n in names if not classify(stats[n], threshold).startswith("flat")]
        if hits:
            print(f"  --> responded: {', '.join(hits)}")
        else:
            print("  --> nothing responded. Every column is within the noise floor.")


if __name__ == "__main__":
    main()
